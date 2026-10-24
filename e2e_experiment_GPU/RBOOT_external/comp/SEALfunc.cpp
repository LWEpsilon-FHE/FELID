#include "SEALfunc.h"
#include <iomanip>

namespace seal{
	void MultipleAdd_SEAL(Evaluator &evaluator, Ciphertext &cipher, Ciphertext& result, long long n) {
		long long k, abs_n;
		long long *binary;
	 
		if(n>=0) abs_n = n;
		else abs_n = -n;

		for(k=1; k<100; k++) {
			if(abs_n < pow2(k)) break;
		}

		binary = new long long[k];
		for(long i=0; i<k; i++) {
			binary[i] = (abs_n / pow2(i)) % 2;
		}

		evaluator.add(cipher, cipher, result);
		if(binary[k-2] == 1) evaluator.add_inplace(result, cipher);

		for(long i=k-3; i>=0; i--) {
			evaluator.add_inplace(result, result);
			if(binary[i] == 1) evaluator.add_inplace(result, cipher);
		}

		if(n<0) evaluator.negate_inplace(result);

	}
	 
	void geneT0T1(Encryptor &encryptor, Evaluator &evaluator, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, Ciphertext& T0, Ciphertext& T1, Ciphertext& cipher)
	{
		double scale = cipher.scale();
		long n = cipher.poly_modulus_degree() / 2;
	 
		vector<double> m_one(n);

		for(int i=0; i<n; i++) m_one[i] = 1.0; 
		Plaintext plain_1;
		encoder.encode(m_one, scale, plain_1);
		Ciphertext ctxt_1;
		encryptor.encrypt(plain_1, ctxt_1);

		T0 = ctxt_1;
		T1 = cipher;

	}
	void evalT(Evaluator &evaluator, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, Ciphertext& Tmplusn, const Ciphertext& Tm, const Ciphertext& Tn, const Ciphertext& Tmminusn)
	{
		Ciphertext temp;
		evaluator.multiply_reduced_error(Tm, Tn, relin_keys, temp);
		evaluator.add_inplace_reduced_error(temp, temp); 
		evaluator.rescale_to_next_inplace(temp); 
		evaluator.sub_reduced_error(temp, Tmminusn, Tmplusn); 
	}

	string to_string(RR x) {
		long old_prec = RR::OutputPrecision();
		RR::SetOutputPrecision(300);
		stringstream ss;
		ss << x;
		RR::SetOutputPrecision(old_prec);
		return ss.str();
	}

#include <cassert>
#include <cstdint>
#include <cmath>
#include <limits>

	RR FixedPoint(double x, int) { return to_RR(x); }
	RR to_fixed_point(RR x, int prec) { return x; }

	void eval_polynomial_integrate(Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, Ciphertext& res, Ciphertext& cipher, long deg, const vector<RR> &decomp_coeff, Tree &tree)
	{
		int prec = 40;
		auto not_cipher = FixedPoint(round(0.984 * 8192) / 8192, prec);
		vector<RR> t_not_cph(deg + 10);
		vector<RR> pt_not_cph(deg + 10);
		vector<string> t_trace(deg + 10);
		vector<string> pt_trace(deg + 10);

		double scale = cipher.scale();		 
		long n = cipher.poly_modulus_degree() / 2;
		long total_depth = ceil_to_int(log(static_cast<double>(deg+1))/log(2.0));		 
		Ciphertext temp1, temp2, state, ctxt_zero;
		evaltype eval_type = tree.type;
		vector<long> decomp_deg(pow2(tree.depth+1), -1);
		vector<long> start_index(pow2(tree.depth+1), -1);
		vector<std::unique_ptr<Ciphertext>> T(deg + 10);
		vector<std::unique_ptr<Ciphertext>> pt(deg + 10);
		for(auto &t_i: T) t_i = nullptr;
		for(auto &pt_i: pt) pt_i = nullptr;
		T[0] = std::make_unique<Ciphertext>();
		T[1] = std::make_unique<Ciphertext>();

		vector<double> m_coeff(n), m_zero(n, 0.0);
		Plaintext plain_coeff, plain_zero;
		encoder.encode(m_zero, scale*scale, plain_zero);		 
		encryptor.encrypt(plain_zero, ctxt_zero);

		long num = 0, temp_index;
		if(eval_type == evaltype::oddbaby) temp_index = 1;
		else if(eval_type == evaltype::baby) temp_index = 0;

		decomp_deg[1] = deg;
		for(int i=1; i<=tree.depth; i++)
		{
			for(int j=pow2(i); j<pow2(i+1); j++)
			{
				if(j>=static_cast<int>(decomp_deg.size())) throw std::invalid_argument("invalid index");
				if(j%2 == 0) decomp_deg[j] = tree.tree[j/2] - 1;
				else if(j%2 == 1) decomp_deg[j] = decomp_deg[j/2] - tree.tree[j/2];
			}
		}

		for(int i=1; i<pow2(tree.depth+1); i++)
		{
			if(tree.tree[i] == 0)
			{
				start_index[i] = temp_index;
				temp_index += (decomp_deg[i]+1);
			}
		}

		geneT0T1(encryptor, evaluator, encoder, public_key, secret_key, relin_keys, *T[0], *T[1], cipher);
		t_not_cph[0] = FixedPoint(1.0, prec);
		t_not_cph[1] = not_cipher;
		t_trace[0] = string("(R(1.0))");
		t_trace[1] = string("(x)");

		if(eval_type == evaltype::oddbaby)
		{
			 
			for(int i=1; i<= total_depth; i++)
			{

				for(int j=1; j<pow2(tree.depth+1); j++)
				{
					if(tree.tree[j] == 0 && total_depth + 1 - num_one(j) == i) 	 
					{
						int temp_idx = start_index[j];

						pt[j] = std::make_unique<Ciphertext>();
						evaluator.multiply_const(*T[1], decomp_coeff[temp_idx], *pt[j]);
						pt_not_cph[j] = t_not_cph[1] * to_fixed_point(decomp_coeff[temp_idx], prec);
						pt_trace[j] = string("(") + t_trace[1] + " * R(" + to_string(decomp_coeff[temp_idx]) + "))";
						temp_idx += 2;
						for(int k=3; k<=decomp_deg[j]; k+=2)
						{
							evaluator.multiply_const(*T[k], decomp_coeff[temp_idx], temp1);
							evaluator.add_inplace_reduced_error(*pt[j], temp1);		 
							pt_not_cph[j] += t_not_cph[k] * to_fixed_point(decomp_coeff[temp_idx], prec);
							pt_trace[j] = string("(") + pt_trace[j] + " + " + t_trace[k] + " * R(" + to_string(decomp_coeff[temp_idx]) + "))";

							temp_idx += 2;
						}
						evaluator.rescale_to_next_inplace(*pt[j]);

					}
				}	

				long inter[40];
				long inter_num = 0;

				for(int j=1; j<pow2(tree.depth+1); j++)
				{
					if(tree.tree[j] > 0 && total_depth + 1 - num_one(j) == i && j%2 == 1) 	 
					{
						long k = j;	
						 
						pt[j] = std::make_unique<Ciphertext>();
						evaluator.multiply_reduced_error(*T[tree.tree[k]], *pt[2*k+1], relin_keys, *pt[j]);
						pt_not_cph[j] = t_not_cph[tree.tree[k]] * pt_not_cph[2*k+1];
						pt_trace[j] = string("(") + t_trace[tree.tree[k]] + " * " + pt_trace[2*k+1] + ")";
						k*=2;
						while(1)
						{
							if(tree.tree[k]==0) break;
							evaluator.multiply_reduced_error(*T[tree.tree[k]], *pt[2*k+1], relin_keys, temp1);
							evaluator.add_inplace_reduced_error(*pt[j], temp1);		 
							pt_not_cph[j] += t_not_cph[tree.tree[k]] * pt_not_cph[2*k+1];
							pt_trace[j] = string("(") + pt_trace[j] + " + " + t_trace[tree.tree[k]] + " * " + pt_trace[2*k+1] + ")";
							k*=2;
						}
						evaluator.rescale_to_next_inplace(*pt[j]);
						evaluator.add_inplace_reduced_error(*pt[j], *pt[k]);
						pt_not_cph[j] += pt_not_cph[k];
						pt_trace[j] = string("(") + pt_trace[j] + " + " + pt_trace[k] + ")";

					}
				}

				if(i<=tree.m-1) 
				{
					 
					T[pow2(i)] = std::make_unique<Ciphertext>();
					evalT(evaluator, public_key, secret_key, relin_keys, *T[pow2(i)], *T[pow2(i-1)], *T[pow2(i-1)], *T[0]);
					t_not_cph[pow2(i)] = t_not_cph[pow2(i-1)] * t_not_cph[pow2(i-1)] * FixedPoint(2.0, prec) - t_not_cph[0];
					t_trace[pow2(i)] = string("(") + t_trace[pow2(i-1)] + "^2 * R(2.0) - " + t_trace[0] + ")";

				}
				
				if(i<=tree.l)
				{
					for(int j=pow2(i-1)+1; j<=pow2(i)-1; j+=2)		 
					{
						 
						T[j] = std::make_unique<Ciphertext>();
						evalT(evaluator, public_key, secret_key, relin_keys, *T[j], *T[pow2(i-1)], *T[j-pow2(i-1)], *T[pow2(i)-j]);
						t_not_cph[j] = t_not_cph[pow2(i-1)] * t_not_cph[j-pow2(i-1)] * FixedPoint(2.0, prec) - t_not_cph[pow2(i)-j];
						t_trace[j] = string("(") + t_trace[pow2(i-1)] + " * " + t_trace[j-pow2(i-1)] + " * R(2.0) - " + t_trace[pow2(i)-j] + ")";

					}
				}

			}
			res = *pt[1];

		}
		else if(eval_type == evaltype::baby)
		{
			 
			for(int i=1; i<= total_depth; i++)
			{

				for(int j=1; j<pow2(tree.depth+1); j++)
				{
					if(tree.tree[j] == 0 && total_depth + 1 - num_one(j) == i) 	 
					{
						int temp_idx = start_index[j];
						 
						pt[j] = std::make_unique<Ciphertext>();

						*pt[j] = ctxt_zero;

						for(int k=0; k<=decomp_deg[j]; k++)
						{
							 
							if(abs(decomp_coeff[temp_idx]) > 1.0/scale)		 
							{
								if(T[k] == nullptr) throw std::runtime_error("T[k] is not set");
								evaluator.multiply_const(*T[k], decomp_coeff[temp_idx], temp1);
								evaluator.add_inplace(*pt[j], temp1);		 
								pt_not_cph[j] += t_not_cph[k] * to_fixed_point(decomp_coeff[temp_idx], prec);
								pt_trace[j] = string("(") + pt_trace[j] + " + " + t_trace[k] + " * R(" + to_string(decomp_coeff[temp_idx]) + "))";
							}
							temp_idx++;
						}
						evaluator.rescale_to_next_inplace(*pt[j]);
						 
					}
				}

				long inter[40];
				long inter_num = 0;

				for(int j=1; j<pow2(tree.depth+1); j++)
				{
					if(tree.tree[j] > 0 && total_depth + 1 - num_one(j) == i) 	 
					{
						int temp = j;
						bool no_execute = false;
						for(int k=0; k<inter_num; k++)
						{
							while(1)
							{
								if(temp == inter[k]){
									no_execute = true;
									break;
								} 
								if(temp%2 == 0) temp/=2;
								else break;
							}
						}
						
						if(no_execute == false)
						{
							inter[inter_num] = j;
							inter_num += 1;

							long k = j;

							pt[j] = std::make_unique<Ciphertext>();
							if(T[tree.tree[k]] == nullptr) throw std::runtime_error("T[tree.tree[k]] is not set");
							if(pt[2*k+1] == nullptr) throw std::runtime_error("pt[2*k+1] is not set");
							evaluator.multiply_reduced_error(*T[tree.tree[k]], *pt[2*k+1], relin_keys, *pt[j]);
							pt_not_cph[j] = t_not_cph[tree.tree[k]] * pt_not_cph[2*k+1];
							pt_trace[j] = string("(") + t_trace[tree.tree[k]] + " * " + pt_trace[2*k+1] + ")";
							k*=2;

							while(1)
							{
								if(tree.tree[k]==0) break;
								if(T[tree.tree[k]] == nullptr) throw std::runtime_error("T[tree.tree[k]] is not set");
								if(pt[2*k+1] == nullptr) throw std::runtime_error("pt[2*k+1] is not set");
								evaluator.multiply_reduced_error(*T[tree.tree[k]], *pt[2*k+1], relin_keys, temp1);
								evaluator.add_inplace(*pt[j], temp1);		 
								pt_not_cph[j] += t_not_cph[tree.tree[k]] * pt_not_cph[2*k+1];
								pt_trace[j] = string("(") + pt_trace[j] + " + " + t_trace[tree.tree[k]] + " * " + pt_trace[2*k+1] + ")";
								k*=2;
							}
							evaluator.rescale_to_next_inplace(*pt[j]);
							evaluator.add_inplace_reduced_error(*pt[j], *pt[k]);
							pt_not_cph[j] += pt_not_cph[k];
							pt_trace[j] = string("(") + pt_trace[j] + " + " + pt_trace[k] + ")";

						}
					}
				}

				for(int j=2; j<=tree.b; j++)
				{
					int g = j;
					if(pow2(i-1)<g && g<=pow2(i))
					{
						 
						T[g] = std::make_unique<Ciphertext>();
						if(g%2 == 0) 
						{
							if(T[g/2] == nullptr) throw std::runtime_error("T[g/2] is not set");
							if(T[0] == nullptr) throw std::runtime_error("T[0] is not set");
							evalT(evaluator, public_key, secret_key, relin_keys, *T[g], *T[g/2], *T[g/2], *T[0]);
							t_not_cph[g] = t_not_cph[g/2] * t_not_cph[g/2] * FixedPoint(2.0, prec) - t_not_cph[0];
							t_trace[g] = string("(") + t_trace[g/2] + "^2 * R(2.0) - " + t_trace[0] + ")";
						}
						else
						{
							if(T[g/2] == nullptr) throw std::runtime_error("T[g/2] is not set");
							if(T[(g+1)/2] == nullptr) throw std::runtime_error("T[(g+1)/2] is not set");
							if(T[0] == nullptr) throw std::runtime_error("T[0] is not set");
							evalT(evaluator, public_key, secret_key, relin_keys, *T[g], *T[g/2], *T[(g+1)/2], *T[1]);
							t_not_cph[g] = t_not_cph[g/2] * t_not_cph[(g+1)/2] * FixedPoint(2.0, prec) - t_not_cph[1];
							t_trace[g] = string("(") + t_trace[g/2] + " * " + t_trace[(g+1)/2] + " * R(2.0) - " + t_trace[1] + ")";
						}
						 
					}
				}
				for(int j=1; j<=tree.m-1; j++)
				{
					int g = pow2(j)*tree.b;
					if(pow2(i-1)<g && g<=pow2(i))
					{
						 
						T[g] = std::make_unique<Ciphertext>();
						if(g%2 == 0) 
						{
							if(T[g/2] == nullptr) throw std::runtime_error("T[g/2] is not set");
							if(T[0] == nullptr) throw std::runtime_error("T[0] is not set");
							evalT(evaluator, public_key, secret_key, relin_keys, *T[g], *T[g/2], *T[g/2], *T[0]);
							t_not_cph[g] = t_not_cph[g/2] * t_not_cph[g/2] * FixedPoint(2.0, prec) - t_not_cph[0];
							t_trace[g] = string("(") + t_trace[g/2] + "^2 * R(2.0) - " + t_trace[0] + ")";
						}
						else
						{
							if(T[g/2] == nullptr) throw std::runtime_error("T[g/2] is not set");
							if(T[(g+1)/2] == nullptr) throw std::runtime_error("T[(g+1)/2] is not set");
							if(T[0] == nullptr) throw std::runtime_error("T[0] is not set");
							evalT(evaluator, public_key, secret_key, relin_keys, *T[g], *T[g/2], *T[(g+1)/2], *T[1]);
							t_not_cph[g] = t_not_cph[g/2] * t_not_cph[(g+1)/2] * FixedPoint(2.0, prec) - t_not_cph[1];
							t_trace[g] = string("(") + t_trace[g/2] + " * " + t_trace[(g+1)/2] + " * R(2.0) - " + t_trace[1] + ")";
						}
						 
					}
				}

			}
			res = *pt[1];

		}
	}
	long coeff_number(long deg, Tree& tree) 
	{

		long num = 0;
		long* decomp_deg = new long[pow2(tree.depth+1)];
		decomp_deg[1] = deg;
		for(int i=1; i<=tree.depth; i++)
		{
			for(int j=pow2(i); j<pow2(i+1); j++)
			{
				if(j%2 == 0) decomp_deg[j] = tree.tree[j/2] - 1;
				else if(j%2 == 1) decomp_deg[j] = decomp_deg[j/2] - tree.tree[j/2];
			}
		}

		for(int i=0; i<pow2(tree.depth+1); i++)
		{
			if(tree.tree[i] == 0) 
			{
				num += (decomp_deg[i]+1);
			}
		}
		delete decomp_deg;
		return num;
	}
	long ShowFailure_ReLU(Decryptor &decryptor, CKKSEncoder &encoder, Ciphertext& cipher, vector<double>& x, long precision, long n) 
	{
		long failure = 0;
		double bound = pow(2.0,static_cast<double>(-precision));
		Plaintext plain_out;
		vector<double> output;
		decryptor.decrypt(cipher, plain_out);
		encoder.decode(plain_out, output);

		for (int i = 0; i < n; ++i) if(abs(ReLU(x[i]) - output[i]) > bound) failure++;

		return failure;

	}
}
