#pragma once

#include <map>
#include <string>

struct Point {
	int x;
	int y;

	friend bool operator<(const Point &p1, const Point &p2)
	{
		if (p1.x == p2.x) {
			return p1.y < p2.y;
		} else {
			return p1.x < p2.x;
		}
	}
};

struct Likelihood {
	float class1;
	float class2;
};

typedef std::map<Point, Likelihood> TrainData;

class Classifier {
public:
	virtual void initialize(const TrainData &data) = 0;
	virtual void step() = 0;

	virtual void classify(const Point &point, Likelihood &l) = 0;
};

typedef std::map<std::string, Classifier *> Classifiers;

