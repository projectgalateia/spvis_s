#if _WIN32
#include <Windows.h>
#endif // _WIN32

#include <GL/gl.h>
#include <GL/glut.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <omp.h>

#include <map>
#include <vector>
#include <thread>
#include <mutex>
#include <iostream>

#include "spvis.hpp"

static std::mutex thread_mutex;

static TrainData points;
static Classifier *classifier = NULL;

Classifiers &getClassifiers()
{
	static Classifiers c;

	return c;
}

void registerClassifier(const std::string &name, Classifier *c)
{
	auto &cs = getClassifiers();

	cs[name] = c;
}

static int window_w;
static int window_h;

struct PointColor {
	Point point;
	float rgb[3];
};

static void display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);

	thread_mutex.lock();
	glPointSize(1);

	std::vector<std::vector<PointColor>> colors;

#pragma omp parallel 
	{
		const int thread_num = omp_get_thread_num();

#pragma omp master
		{
			colors.resize(omp_get_num_threads());
		}
#pragma omp barrier

#pragma omp for
		for (int i = 0; i < window_h; ++i) {
			for (int j = 0; j < window_w; ++j) {
				const Point p = {j, i};
				Likelihood l;

				classifier->classify(p, l);
				l.normalize();

				float d = std::abs(l.class1 - l.class2);
				float c = 1 - d;

				float r = l.class1 + (1 - l.class1) * c;
				float g = c;
				float b = l.class2 + (1 - l.class2) * c;

				colors[thread_num].push_back({p, {r, g, b}});
			}
		}
	}
	thread_mutex.unlock();

	for (const auto &pcr : colors) {
		for (const auto &pc : pcr) {
			glColor3fv(pc.rgb);

			glBegin(GL_POINTS);
			glVertex2iv((int *)&pc.point);
			glEnd();
		}
	}

	glPointSize(7);
	glColor3f(0.0f, 0.0f, 0.0f);
	glBegin(GL_POINTS);
	for (const auto &v : points) {
		glVertex2i(v.first.x, v.first.y);
	}
	glEnd();

	glPointSize(3);
	for (const auto &v : points) {
		glColor3f(v.second.class1, 0.0f, v.second.class2);

		glBegin(GL_POINTS);
		glVertex2i(v.first.x, v.first.y);
		glEnd();
	}

	glutSwapBuffers();
}

static void reshape(int w, int h)
{
	window_w = w;
	window_h = h;

	printf("%d %d\n", w, h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluOrtho2D(0, w, h, 0);
	glViewport(0, 0, w, h);
}

static void keyboard(unsigned char key, int x, int y)
{
	if (key == 27 || key == 'q') {
		exit(0);
	}

	if (key == 'r') {
		points.clear();
		thread_mutex.lock();
		classifier->initialize(points);
		thread_mutex.unlock();
	}
}

static void special(int key, int x, int y)
{
}

static void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_UP)
		return;

	Point p = {x, y};

	points[p] = {(button == GLUT_LEFT_BUTTON ? 1.0f : 0.0f), (button == GLUT_RIGHT_BUTTON ? 1.0f : 0.0f)};

	thread_mutex.lock();
	classifier->initialize(points);
	thread_mutex.unlock();
}

int main(int argc, char **argv)
{
	using namespace std;

	const auto &classifiers = getClassifiers();

	if (classifiers.size() == 0) {
		cerr << "No Classifier compiled" << endl;
		return 1;
	}

	int idx = -1;
	std::vector<Classifier *> cvec;

	cout << "Supported Classifiers:" << endl;

	for (const auto &c : classifiers) {
		cout << cvec.size() << "\t: " << c.first << endl;
		cvec.push_back(c.second);
	}

	cout << "Select Classfier: ";

	while (idx < 0 || idx > classifiers.size()) {
		cin >> idx;
	}

	classifier = cvec[idx];

	std::thread step_thread([]()
		{
		while (true) {
		thread_mutex.lock();
		classifier->step();
		thread_mutex.unlock();
		}
		});

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

	glutCreateWindow("spvis");

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutMouseFunc(mouse);
	glutIdleFunc(display);

	glClearColor(0.0, 0.0, 0.0, 1.0);

	glutMainLoop();

	return 0;
}

