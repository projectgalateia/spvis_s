#include "spvis.hpp"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#include <cblas.h>

static class MLP : Classifier {
private:	
	template<size_t M, size_t N>
	struct Layer {
		const float dt = 0.5f;

		float W[N][M];
		float A[N];
		float B[N];
		float H[N];
		
		float dW[N][M];
		float dA[N];
		float dH[N];

		Layer()
		{
			reset();
		}

		void reset()
		{
			memset(B, 0x0, sizeof(B));
			
			for (int i = 0; i < N; ++i) {
				for (int j = 0; j < M; ++j) {
					W[i][j] = float(rand()) / float(RAND_MAX);
				}
			}
		}

		float step_function(float v)
		{
			return 1.0f / (1.0f + exp(-v));
		}

		template<size_t K>
		void propagate(Layer<K, M> &l)
		{
			cblas_scopy(N, B, 1, A, 1);
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				N, 1, M, 1.0f, (float *)W, M, l.H, 1, 1.0f, A, 1);

			for (int i = 0; i < N; ++i) {
				H[i] = step_function(A[i]);
			}
		}

		template<size_t K>
		void backpropagate(Layer<K, M> &l)
		{
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				N, M, 1, 1.0f, dA, 1, l.H, M, 0.0f, (float *)dW, M); 

			cblas_saxpy(N, dt, dA, 1, B, 1);
			
			cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				M, 1, N, 1.0f, (float *)W, M, dA, 1, 0.0f, l.dH, 1);

			for (int i = 0; i < M; ++i) {
				const float g = step_function(l.A[i]);

				l.dA[i] = g * (1 - g) * dH[i];
			}

			cblas_saxpy(M * N, dt, (float *)dW, 1, (float *)W, 1);
		}

		void setOutput(float O[N])
		{
			for (int i = 0; i < N; ++i) {
				H[i] = O[i];
			}
		}

		void makeError(float E[N])
		{
			for (int i = 0; i < N; ++i) {
				dA[i] = E[i] - H[i];
			}
		}
	};

	TrainData data;
	Layer<1,  2> input;
	Layer<2, 10> l1;
	Layer<10,15> l2;
	Layer<15, 2> output;

	void calc(const Point &p, Likelihood &l)
	{
		int w, h;

		screen_prop(w, h);

		float inp[2];

		inp[0] = float(p.x) / float(w);
		inp[1] = float(p.y) / float(h);

		input.setOutput(inp);
		l1.propagate(input);
		l2.propagate(l1);
		output.propagate(l2);

		l.class1 = output.H[0] / (output.H[0] + output.H[1]);
		l.class2 = output.H[1] / (output.H[0] + output.H[1]);
	}

public:
	MLP()
	{
		registerClassifier("MLP", this);
	}

	void initialize(const TrainData &data, const TrainVector &vector) 
	{
		input.reset();
		l1.reset();
		l2.reset();
		output.reset();

		this->data = data;
	}

	void step()
	{
		float err = 0.0f;
		
		float e[2];

		for (const auto &d : data) {
			Likelihood l;

			calc(d.first, l);

			e[0] = d.second.class1;
			e[1] = d.second.class2;

			output.makeError(e);
			output.backpropagate(l2);
			l2.backpropagate(l1);
			l1.backpropagate(input);

			err += pow(output.dA[0], 2) + pow(output.dA[1], 2);
		}

		err /= data.size();

		//printf("Error: %f\n", err);
	}

	void classify(const Point &point, Likelihood &l)
	{
		calc(point, l);
	}
} classifier;
