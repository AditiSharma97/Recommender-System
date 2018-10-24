#! /usr/bin/python3
import os.path
import math
import numpy
import heapq
import time

class RecommenderSystems:
	
	def __init__ (self):
		print ("in constructor")

	def loadData (self):
		#File containing ratings
		ratings_file = open('./ratings/ratings.dat', 'r')
		self.ratings_matrix = numpy.zeros ((3952, 6040))
		#print (self.ratings_matrix)
	
		for rating in ratings_file:
			data = rating.strip().split('::')
			#print (data)
			self.ratings_matrix[int(data[1]) - 1, int(data[0]) - 1] = float (data[2])
			#print (ratings_matrix)
	
		ratings_file.close()

	def preprocessing (self):
		self.transpose_ratings_matrix = numpy.transpose (self.ratings_matrix)
		self.total_mean = self.ratings_matrix.mean()
		
		self.movie_sums = self.ratings_matrix.sum(1)
		self.movie_non_zeros = numpy.count_nonzero(self.ratings_matrix, 1)
		self.movie_means = numpy.zeros(self.movie_sums.shape)
		
		for i in range (self.movie_sums.size):
			if self.movie_non_zeros[i] != 0:
				self.movie_means[i] = self.movie_sums[i] / self.movie_non_zeros[i]

		self.user_sums = self.transpose_ratings_matrix.sum(1)
		self.user_non_zeros = numpy.count_nonzero(self.transpose_ratings_matrix, 1)
		self.user_means = numpy.zeros(self.user_sums.shape)

		for i in range (self.user_sums.size):
			if self.user_non_zeros[i] != 0:
				self.user_means[i] = self.user_sums[i] / self.user_non_zeros[i]

		
	def preprocessCollaborativeFiltering (self):
		magnitudes = numpy.zeros (self.movie_sums.shape)
		norm_ratings_matrix = numpy.zeros(self.ratings_matrix.shape)
	
		for i in range (self.ratings_matrix.shape[0]):
			for j in range (self.ratings_matrix.shape[1]):
				if self.ratings_matrix [i, j] != 0:
					norm_ratings_matrix [i, j] = (self.ratings_matrix [i, j] - self.movie_means[i])
					magnitudes[i] += norm_ratings_matrix [i, j] * norm_ratings_matrix [i, j]
		
			magnitudes[i] = math.sqrt(magnitudes[i])
			
			for j in range (self.ratings_matrix.shape[1]):
				if magnitudes[i] != 0.0:
					norm_ratings_matrix [i, j] = norm_ratings_matrix [i, j] / magnitudes[i]

		similarities = numpy.zeros((self.ratings_matrix.shape[0], self.ratings_matrix.shape[0]))
		most_similar = numpy.zeros((self.ratings_matrix.shape[0], 100))

		most_similar_file = open('./ratings/most_similar.dat', 'w')
		print ("opened")

		for i in range(self.ratings_matrix.shape[0]):
			for j in range(i + 1, norm_ratings_matrix.shape[0]):
				for k in range(norm_ratings_matrix.shape[1]):
					similarities[i][j] += norm_ratings_matrix[i,k] * norm_ratings_matrix[j,k]				
			similarities[j][i] = similarities[i][j]
			most_similar[i] = heapq.nlargest(100, range(len(similarities[i])), similarities[i].take)
			if i is 0:
				print (most_similar[i])
			for j in range(0, 100):
				write_string = str(i) + '::' + str(j) + '::' + str(int(most_similar[i][j])) + '::' + str(similarities[i][int(most_similar[i][j])]) + '\n'
				most_similar_file.write(write_string)

		most_similar_file.close()

	def colab(self):

		start_time = time.time()

		# If preprocessed file doesn't exist, create it
		if not os.path.exists('./ratings/most_similar.dat'):
			self.preprocessCollaborativeFiltering()

		# Load the preprocessed file
		most_similar_file = open('./ratings/most_similar.dat', 'r')

		similarities = numpy.zeros((self.ratings_matrix.shape[0], self.ratings_matrix.shape[0]))
		top_100 = {}

		for line in most_similar_file:
			data = line.strip().split('::')
			i = int(data[0])
			j = int(data[2])
			s = data[3]
		    
			if i not in top_100:
				top_100[i] = []
			top_100[i].append(j)
			similarities[i][j] = s

		most_similar_file.close()

		guesses = numpy.zeros((400, 400))
		baseline_guesses = numpy.zeros((400, 400))

		rmse = 0
		baseline_rmse = 0
		values_tested = 0

		for i in range(400):
			for j in range(400):
				if self.ratings_matrix[i][j] != 0:
					values_tested += 1
					simsum = 0.0
					for k in top_100[i]:
						if self.ratings_matrix[k][j] != 0:
							guesses[i][j] += self.ratings_matrix[k][j] * similarities[i][k]
							baseline_score = self.total_mean + self.movie_means[k] + self.user_means[j]
							baseline_guesses[i][j] += (self.ratings_matrix[k][j] - baseline_score) * similarities[i][k]
							simsum += similarities[i][k]
					if simsum != 0.0:
						guesses[i][j] /= simsum
						baseline_guesses[i][j] /= simsum
					baseline_guesses[i][j] += self.total_mean + self.movie_means[i] + self.user_means[j]
					guess_error = guesses[i][j] - self.ratings_matrix[i][j]
					rmse += guess_error ** 2
					baseline_guess_error = baseline_guesses[i][j] - self.ratings_matrix[i][j]
					baseline_rmse += baseline_guess_error ** 2

		rmse /= values_tested
		rho = 1 - (6 / (values_tested ** 2 - 1)) * rmse
		rmse = math.sqrt(rmse)
		baseline_rmse /= values_tested
		baseline_rho = 1 - (6 / (values_tested ** 2 - 1)) * baseline_rmse
		baseline_rmse = math.sqrt(baseline_rmse)

		t_guesses = numpy.transpose(guesses)
		t_baseline_guesses = numpy.transpose(baseline_guesses)

		precision_at_10 = 0.0
		baseline_precision_at_10 = 0.0

		for i in range(400):
			top_10 = heapq.nlargest(10, range(400), t_guesses[i].take)
			for r in top_10:
				if r >= 3.5:
					precision_at_10 += 1
			top_10 = heapq.nlargest(10, range(400), t_baseline_guesses[i].take)
			for r in top_10:
				if r >= 3.5:
					baseline_precision_at_10 += 1

		precision_at_10 /= 4000
		baseline_precision_at_10 /= 4000

		print ('Collaborative\t\t')

		print ('{0:.3f}'.format(rmse) + '\t')

		print ('{0:.3f}'.format(precision_at_10 * 100) + '%\t\t')

		print ('{0:.6f}'.format(rho) + '\t\t\t')

		print ('{0:.3f}'.format((time.time() - start_time) / 60) + '\n\n')

		print ('Baseline Collaborative\t')

		print ('{0:.3f}'.format(baseline_rmse) + '\t')

		print ('{0:.3f}'.format(baseline_precision_at_10 * 100) + '%\t\t')

		print ('{0:.6f}'.format(baseline_rho) + '\n\n')

def main():
	obj = RecommenderSystems()
	obj.loadData()
	obj.preprocessing()
	obj.colab()

if __name__ == '__main__':
	main()
