#include "spvis.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

static class NBC : Classifier {
private:
	float mean[2][2];
	float V[2][2];

public:
	NBC()
	{
		registerClassifier("NBC", this);
	}

	void initialize(const TrainData &data)
	{
		std::vector<Point> vecs[2];

		memset(mean, 0x0, sizeof(mean));
		memset(V, 0x0, sizeof(V));

		for (const auto &d : data) {
			if (d.second.class1 > 0) {
				vecs[0].push_back(d.first);
			} else {
				vecs[1].push_back(d.first);
			}
		}

#pragma omp parallel for reduction(+:mean[0][0], +:mean[0][1])
		for (int i = 0; i < vecs[0].size(); ++i) {
			mean[0][0] += vecs[0][i].x;
			mean[0][1] += vecs[0][i].y;
		}

#pragma omp parallel for reduction(+:mean[1][0], +:mean[1][1])
		for (int i = 0; i < vecs[1].size(); ++i) {
			mean[1][0] += vecs[1][i].x;
			mean[1][1] += vecs[1][i].y;
		}

		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 2; ++j) {
				mean[i][j] /= vecs[i].size();
			}
		}

#pragma omp parallel for reduction(+:V[0][0], +:V[0][1])
		for (int i = 0; i < vecs[0].size(); ++i) {
			V[0][0] += pow(vecs[0][i].x - mean[0][0], 2);
			V[0][1] += pow(vecs[0][i].y - mean[0][1], 2);
		}

#pragma omp parallel for reduction(+:V[1][0], +:V[1][1])
		for (int i = 0; i < vecs[1].size(); ++i) {
			V[1][0] += pow(vecs[1][i].x - mean[1][0], 2);
			V[1][1] += pow(vecs[1][i].y - mean[1][1], 2);
		}
		
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 2; ++j) {
				V[i][j] /= vecs[i].size();
			}
		}
	}

	void step()
	{
	}

	void classify(const Point &point, Likelihood &l)
	{
		l.class1 = exp(-pow(point.x - mean[0][0], 2.0f) / (2*V[0][0]))
			* exp(-pow(point.y - mean[0][1], 2.0f) / (2*V[0][1]));
		l.class2 = exp(-pow(point.x - mean[1][0], 2.0f) / (2*V[1][0]))
			* exp(-pow(point.y - mean[1][1], 2.0f) / (2*V[1][1]));
	}
} classifier;

