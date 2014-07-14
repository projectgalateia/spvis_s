#if _WIN32
#include <Windows.h>
#endif // _WIN32

#include <GL/gl.h>
#include <GL/glut.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <map>
#include <thread>
#include <mutex>

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

static void display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);

	thread_mutex.lock();
	glPointSize(1);
	for (int i = 0; i < window_h; ++i) {
		for (int j = 0; j < window_w; ++j) {
			const Point p = {j, i};
			Likelihood l;

			classifier->classify(p, l);
			l.normalize();

			float d = std::abs(l.class1 - l.class2);
			float c = 1 - d;

			glColor3f(l.class1 + (1 - l.class1) * c, c, l.class2 + (1 - l.class2) * c);

			glBegin(GL_POINTS);
			glVertex2i(j, i);
			glEnd();
		}
	}
	thread_mutex.unlock();

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
	const auto &classifiers = getClassifiers();

	if (classifiers.size() == 0) {
		fprintf(stderr, "No Classifier compiled\n");
		return 1;
	}

	classifier = classifiers.begin()->second;

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

