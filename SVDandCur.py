import numpy as np
import random
import time
from operator import itemgetter
import scipy

# === Recommeder System using Collabarative filtering ===


class svdAndCur:


"""
Class  svdAndCur , contains methods to decompose a matrix by SVD or CUR algorithms.

In simple words , implement CUR and SVD algorithms.
"""

	def __init__(self):
		print("in constructor")

	def precision(self):
"""
Precision method calculates the precision between the predicted values and intial data

	param:

	return:
"""
		// taking top 10 predictions
		ratings_file = open('./ratings.dat', 'r')

		user = {}
		precision = 0.0
		ultra_count = 0

		for rating in ratings_file:
			// print("printing the: ", rating, "\n")
			predict1 = {} // saving hte eal matrix with movie keys
			predict2 = {} // ssaving the predicted ratings with movie keys
			if(len(user) == 201):
				break
			else:
				pass

			data = rating.strip().split('::')
			if not data[0] in user:
				user[data[0]] = True

			prv_user = int(data[0]) - 1
			prv_user_rating = float(data[2])
			prv_user_movie = int(data[1]) - 1

			if prv_user_movie in predict1:
				pass
			else:
				predict1[prv_user_movie] = prv_user_rating

			if prv_user_movie in predict2:
				pass
			else:
				predict2[prv_user_movie] = self.ratings_matrix[prv_user][prv_user_movie]

			for rating in ratings_file:
				new_data = rating.strip().split("::")
				current_user = int(new_data[0]) - 1
				// print("in while printing the : ", rating, " ", prv_user, " ", current_user)
				current_user_rating = float(new_data[2])
				current_user_movie = int(new_data[1]) - 1

				if prv_user != current_user:
					myList = sorted(predict2.items(), key=itemgetter(1), reverse=True)
					count = 0

					for temp in range(10):
						if(predict1[myList[temp][0]] >= 3.5): // threshhold in 3.5
							count += 1;
						else:
							pass
					print("count : ", count / 10)
					ultra_count += 1
					precision += (count / 10)

					break
				else:
					pass

				if current_user_movie in predict1:
					pass
				else:
					predict1[current_user_movie] = current_user_rating

				if current_user_movie in predict2:
					pass
				else:
					predict2[current_user_movie] = self.ratings_matrix[current_user][current_user_movie]

		precision = precision / 201;

		print("\nprecision is : ", precision, " ", ultra_count)

	def spearman_correlation (self) :

"""
Calculates the spearaman Correlation 
	param:
	return:
"""
		print("The spearman_correlation is : " , scipy.stats.spearmanr(self.intial_data , self.ratings_matrix)[0] )


	def rmse(self):
"""
Calculates RMSE betweeen the predicted values and the intial data 
	param:
	return:
"""
		// File containing ratings
		ratings_file = open('./ratings.dat', 'r')

		// Taking 20% of our data as test data
		user = {}
		rmse = 0.0
		count = 0
		for rating in ratings_file:
			if(len(user) == 201):
				break
			else:
				pass

			data = rating.strip().split('::')
			if not data[0] in user:
				user[data[0]] = True

			rmse += (self.ratings_matrix[int(data[0]) - 1, int(data[1]) - 1] - float(data[2])) ** 2
			// print("printing the rmse : " , rmse  , "\n")
			count += 1

		ratings_file.close()
		rmse = np.sqrt(rmse / count)

		print("\n The rmse of is : ", rmse, "\n")

	def loadData(self):
"""
Loading the data from .dat file 

Ratings  format:
		
		UserID::MovieID::Rating::Timestamp

	param:
	return:
"""
		// File containing ratings
		ratings_file = open('./ratings.dat', 'r')
		self.ratings_matrix = np.zeros((3952, 6040))
		// print (self.ratings_matrix)

		for rating in ratings_file:
			// print("\n" , "printing the : " , rating , "\n")
			data = rating.strip().split('::')
			// print (data)
			self.ratings_matrix[int(data[1]) - 1, int(data[0]) - 1] = float(data[2])
			// print (ratings_matrix)

		ratings_file.close()
		print("loading data successful")

		self.ratings_matrix = self.ratings_matrix.T  // 6040 * 3952
		self.ratings_matrix = self.ratings_matrix[: 1002, :]  // taking only 1002 users
		self.intial_data = self.ratings_matrix
		// print (self.ratings_matrix)
		// rating matrix contains movies as rows and users as clumns , each cell has a rating

	def svd_calc(self, dim_reduction):  // A is user-movie rating matrix , dim_reduction is a bool , whether needed to do dim reduction or not
"""
For SVD calculation method, 

	param: self.rating_matrix is the intial data  
		   dim_reduction : whether to decrease the energy
		   to 90 % or not 
	return:
   
"""
		start = time.time()
		// Assumed A dim as m*n , where m <= n , still a check done
		A = self.ratings_matrix
		if (A.shape[0] <= A.shape[1]):
			// flag = 1
			// print("transposing hte A in start ")
			// A = A.T

			AAt = A @ A.T

			// calculating u and s^2  following the equation : A*transpoose(A)*U = S^2 * U
			// U will be an m × m square matrix since there can be at most m non-zero singular values, while V will be an n × m matrix.
			s, u = np.linalg.eig(AAt)

			// sorting to descendnig  form
			idx = np.argsort(-s)  // return type : array ; return array of indixes for sorting  ; "-"  make sure that it's sorted in descending order
			s = s[idx]
			u = u[:, idx]

			// ignoring the imaginarvy parts
			s = s.real  // vector
			u = u.real  // matrix

			// Creating the diagnol matrix from the vector
			sigma = np.diag(np.sqrt(s))

			// calculating v with the  following equation : V = transpose(A) * U * S-1
			v = A.T @ u @ np.linalg.pinv(sigma)
		else:
			AtA = A.T @ A
			s, v = np.linalg.eig(AtA)
			idx = np.argsort(-s)  // return type : array ; return array of indixes for sorting  ; "-"  make sure that it's sorted in descending order
			s = s[idx]
			v = v[:, idx]
			s = s.real
			v = v.real

			sigma = np.diag(np.sqrt(s))

			u = A @ v @ np.linalg.pinv(sigma)

		// doing the dimension reduction with 90% energy
		if dim_reduction:
			energy = s.sum()
			energy = energy * 0.9

			size_s = s.shape[0]
			k = 0

			for i in range(size_s):
				if energy > 0:
					energy -= s[i]
					k += 1
				else:
					break

			u = u[:, :k]
			v = v[:, : k]
			s = s[:k]

			sigma = np.diag(np.sqrt(s))

		print(u.shape, sigma.shape, v.shape)
		print("U matrix : \n", u)
		print("sigma matrix : \n", sigma)
		print("v matrix : \n", v)

		print("\n\n")

		print("original matrix : \n", A)
		self.ratings_matrix = u @ sigma @ v.T
		print("recontructed matrix : \n", self.ratings_matrix)

		//  converting hte sigma to m* n
		// sigma_origimal = np.zeros((A.shape[0], A.shape[1]))
		// sigma_origimal[:sigma.shape[0], : sigma.shape[1]] = sigma

		//  converting V into n*n matrix
		// v_original = np.zeros((A.shape[1], A.shape[1]))
		// v_original[:v.shape[0], :v.shape[1]] = v

		end = time.time()
		print("\nTook " + str(end - start) + " seconds to generate svd\n")
		return u, sigma, v

	def pseudoInverse(self, Z):
"""
return the pseudoInvere of the matrix 

	param :	Z is the data matrix for pseudoinverse
	return: 	the pseudo inverse matrix of Z 

"""

		row = Z.shape[0]
		col = Z.shape[1]

		for i in range(row):
			for j in range(col):
				if Z[i][j] != 0:
					Z[i][j] = 1 / Z[i][j]
				else:
					pass

		return Z

	def cur(self,  rowInR, colInC, dim_reduction):  // rowInr == colInC ( generally)
"""

return the pseudoInvere of the matrix 

	param :	rowInR - numbers of row to be picked for R matrix ;
			colInC - numbers of col to be picked for C matrix :
			dim_reduction - to reduce the enrgy to 90 % or not 
	return: 	
"""

		data = self.ratings_matrix

		dataT = data.T
		col = {}  // dictionary to save the random colums which we are copying
		row = {}  // key is "index" and the value is the "frequency"

		// Creating the C matrix
		// for future reference : https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array
		for temp in range(colInC):  // creating of dictionary
			randCol = random.randint(0, data.shape[1] - 1)

			if randCol in col:
				col[randCol] += 1
			else:
				col[randCol] = 1

		flag = 0
		for temp in col:  // for appending the columns in C
			if (flag == 0):
				flag = 1
				C = data[:, temp].reshape(-1, 1) * np.sqrt(col[temp])
			else:
				C = np.c_[C, np.sqrt(col[temp]) * data[:, temp]]

		// Creaitng R matrix
		for temp in range(rowInR):  // creating the dictionary
			randRow = random.randint(0, data.shape[0] - 1)

			if randRow in row:
				row[randRow] += 1
			else:
				row[randRow] = 1

		flag = 0
		for temp in row:  // appending the row as columns and later transposing hte matrix , imp used dataT matrix instead of data
			if(flag == 0):
				flag = 1
				R = dataT[:, temp].reshape(-1, 1) * np.sqrt(row[temp])
			else:
				R = np.c_[R, np.sqrt(row[temp]) * dataT[:, temp]]

		R = R.T  // transposing , reasons mentioned in above comment

		print("printing hte C matrix : \n", C)
		print("\n")
		print("printing he r matrix : \n", R, "\n")

		print("printing hte col dictionary : \n", col, "\n", " printing the row dictionary : \n", row)

		// Making of W matrix
		W = np.zeros((len(row), len(col)))
		print(W)
		// Filling the W matrix
		i = 0
		for rowIndex in row:
			j = 0
			for colIndex in col:
				W[i][j] = data[rowIndex][colIndex]
				j += 1
			i += 1

		print("printing the W matrix : \n ", W)

		// X, Z, Y = np.linalg.svd(W, full_matrices=False)
		// Z = np.diag(Z)
		// Y = Y.T
		// print("shape of X : \n", X.shape, "\n shape of Z : \n", Z.shape, "\n shape of Y : \n", Y.shape, "\n")
		self.ratings_matrix = W 

		X, Z, Y = self.svd_calc( dim_reduction)
		print("shape of X : \n", X.shape, "\n shape of Z : \n", Z.shape, "\n shape of Y : \n", Y.shape, "\n")
		Zplus = self.pseudoInverse(Z)

		U = Y @ Zplus @ X.T

		print(" the C matrix is : \n ", C, "\nthe U matrix is :\n", U, "\n the r matrix is : \n", R, " \n")

		print("the data is : \n", data)
		print("\n\n")
		self.ratings_matrix = C @ U @ R 
		print("the reconstructed matrix is : \n", self.ratings_matrix )

		return C, U, R



# === SVD and CUR calculation ===


obj = svdAndCur()

obj.loadData()
obj.svd_calc(False)  // svd_calculation woth 100% energy
obj.rmse()
obj.precision() 
obj.spearman_correlation()

obj.loadData()
obj.svd_calc(True) 	// svd_calcilation with 90 % energy

obj.rmse()
obj.precision() 
obj.spearman_correlation()
obj.loadData()
start = time.time()
obj.cur(800 , 4000 , False) 	// svd_calcilation with 90 % energy


end = time.time() 
print("\nTook " + str(end - start) + " seconds to generate svd\n")
obj.rmse()
obj.precision() 
obj.spearman_correlation()
obj.loadData()
start = time.time()
obj.cur(800 , 4000 , True) 	// svd_calcilation with 90 % energy


end = time.time() 
print("\nTook " + str(end - start) + " seconds to generate svd\n")
obj.rmse()
obj.precision() 
obj.spearman_correlation()

