#include <GL/gl.h>
#include <GL/glut.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <map>

#include "spvis.hpp"

static TrainData points;

Classifiers classifiers;

static Classifier *classifier = NULL;

static int window_w;
static int window_h;

static void display()
{
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);

	classifier->step();

	glPointSize(1);
	for (int i = 0; i < window_h; ++i) {
		for (int j = 0; j < window_w; ++j) {
			const Point p = {j, i};
			Likelihood l;

			classifier->classify(p, l);

			float d = std::abs(l.class1 - l.class2);
			float c = 1 - d;

			glColor3f(c + l.class1 * d, c, c + l.class2 * d);

			glBegin(GL_POINTS);
			glVertex2i(j, i);
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
	if (key == 27) {
		exit(0);
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

	classifier->initialize(points);
}

int main(int argc, char **argv)
{
	if (classifiers.size() == 0) {
		fprintf(stderr, "No Classifier compiled\n");
		return 1;
	}

	classifier = classifiers.begin()->second;

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

