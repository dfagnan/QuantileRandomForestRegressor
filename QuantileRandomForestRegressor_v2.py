##python QuantileRandomForestRegressor

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

class QuantileRandomForestRegressor(RandomForestRegressor):
	def __init__(self, n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, min_density=None, compute_importances=None):
		RandomForestRegressor.__init__(self, n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes= max_leaf_nodes, bootstrap= bootstrap, compute_importances=compute_importances, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose)

	def fit(self,X, y):
		self.target_columns = y.columns.tolist()
		RandomForestRegressor.fit(self,X,y)
		leaf_results = pd.DataFrame(self.apply( X))
		#leaf_results['ind'] = 1
		self.observations = pd.concat([leaf_results,y],axis=1) 

						
		
				


		

		

	def predict(self,X, quantiles = [.80,.50,.20], withMean = True):
		
		self.leaf_results = pd.DataFrame(self.apply( X))
		out_q = np.ndarray((len(self.leaf_results),len(self.target_columns),(len(quantiles)+withMean)), dtype=float)

		if withMean:
			mean_predictions = RandomForestRegressor.predict(self,X)

		for obs in range(len(self.leaf_results)):
			self.target_dist_dict = {}
			for target in self.target_columns:
					self.target_dist_dict[target] = pd.DataFrame(columns = ['weight', target])

			for tree in range(self.n_estimators):
				#if tree not in self.target_columns:
				subspace_obs = self.observations[self.observations[tree].astype(int)==self.leaf_results[tree][obs].astype(int)][self.target_columns]
				for target in self.target_columns:	
					subspace_obs[str(target)+'_weight'] = subspace_obs[target].count()
				#if tree%50 == 0:
					#print len(subspace_obs)
					#print subspace_obs.weight
				for target in self.target_columns:
					self.target_dist_dict[target] = self.target_dist_dict[target].append(subspace_obs[[str(target)+'_weight',target]])

			for target in range(len(self.target_columns)):
				t_name = self.target_columns[target]
				t_dist = self.target_dist_dict[t_name].sort(columns=[t_name], ascending=[1], inplace=False)
				t_dist.weight = 1.0/((t_dist[str(t_name)+'_weight']*self.n_estimators+1.0))
				t_dist['cumsum'] = np.cumsum(t_dist.weight)
				for p in range(len(quantiles)):
					#print str(n) +" " + str(target) + " " + str(p)
					out_q[obs,target,p] = t_dist[t_dist['cumsum'] <= quantiles[p]][t_name].max()
				self.test_dist = t_dist
		return out_q








					




		"""
		self.distributions = []
		##iterate through samples
		#add set of distributions for each target for given sample 
		out_q = np.ndarray((len(leaf_results),len(self.target_columns),(len(quantiles)+withMean)), dtype=float)
		mean_predictions = None
		

		for n in range(len(leaf_results)):
			








			
			for target in range(len(self.target_columns)):
				for p in range(len(quantiles)):
					#print str(n) +" " + str(target) + " " + str(p)
					out_q[n,target,p] = self.distributions[n][self.target_columns[target]][self.distributions[n][self.target_columns[target]]['cumsum'] <= quantiles[p]].value.max()

				if withMean:
					out_q[n,target,p+1] = mean_predictions[n,target]

		return out_q
		"""


		






