##python QuantileRandomForestRegressor

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

class QuantileRandomForestRegressor(RandomForestRegressor):
	def __init__(self, n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, min_density=None, compute_importances=None):
		RandomForestRegressor.__init__(self, n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes= max_leaf_nodes, bootstrap= bootstrap, compute_importances=compute_importances, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose)

	def fit(self,X, y):
		self.target_columns = np.array(y.columns)
		num_targets = len(y.iloc[0])
		RandomForestRegressor.fit(self,X,y)
		self.tree_leaf_dicts = []
		leaf_results = pd.DataFrame(self.apply( X))
		#leaf_results['ind'] = 1
		self.tree_weights = []
		#self.leaf_values = [0]*len(leaf_results.columns)
		for t in leaf_results:
			tree = leaf_results[t]
			self.tree_weights.append({})
			self.tree_leaf_dicts.append({})
			leaf_dict = self.tree_leaf_dicts[t]
			for i in range(len(tree)):
				obs = tree[i]
					#iterate through rows of observations on single tree
					#compare observed leaf to dict -if exists append values else add key to dict
				if(obs in leaf_dict):
					for col in self.target_columns:
						leaf_values =leaf_dict[obs][col]
						#leaf # is assigned values
						leaf_dict[obs][col] = np.append(leaf_values,y.iloc[i][col])
				else:
					leaf_dict[obs] = {}
					for col in self.target_columns:
						#leaf # is assigned values
						if(np.isfinite(y.iloc[i][col])):
							leaf_dict[obs][col]=np.array(y.iloc[i][col])

		for tree in range(len(self.tree_leaf_dicts)):
			self.tree_weights.append({})
			for leaf in self.tree_leaf_dicts[tree]:
				self.tree_weights[tree][leaf] = {}
				for col in self.target_columns:
					self.tree_weights[tree][leaf][col] = (np.count_nonzero(~np.isnan(self.tree_leaf_dicts[tree][leaf][col])))

						
		
				


		

		

	def predict(self,X, quantiles = [.80,.50,.20], withMean = True):
		
		leaf_results = pd.DataFrame(self.apply( X))

		self.distributions = []
		##iterate through samples
		#add set of distributions for each target for given sample 
		out_q = np.ndarray((len(leaf_results),len(self.target_columns),(len(quantiles)+withMean)), dtype=float)
		mean_predictions = None
		if withMean:
			mean_predictions = RandomForestRegressor.predict(self,X)

		for n in range(len(leaf_results)):
			self.distributions.append({})
			#iterate through columns in sample (trees) j=sample tree
			for col in self.target_columns:
				self.distributions[n][col]=pd.DataFrame(columns=['weight','value'])
				#leaf_dict[col] = y.iloc[n][col]
				for j in range(self.n_estimators):
					temp_df = pd.DataFrame(columns=['weight','value'])
					#print str(j)+" "+str(n)+" "+str(leaf_results.iloc[n][j])+" "+str(self.tree_leaf_dicts[j][leaf_results.iloc[n][j]][col])
				
					temp_df['value'] = np.atleast_1d(self.tree_leaf_dicts[j][leaf_results.iloc[n][j]][col])
					
					temp_df['weight'] = 1.0/(self.tree_weights[j][leaf_results.iloc[n][j]][col]*(self.n_estimators)+1)
					self.distributions[n][col]= self.distributions[n][col].append(temp_df)

				
				self.distributions[n][col].sort(columns=['value'],inplace=True, ascending=[1])
				self.distributions[n][col]['cumsum'] = np.cumsum(self.distributions[n][col].weight)

			
			for target in range(len(self.target_columns)):
				for p in range(len(quantiles)):
					#print str(n) +" " + str(target) + " " + str(p)
					out_q[n,target,p] = self.distributions[n][self.target_columns[target]][self.distributions[n][self.target_columns[target]]['cumsum'] <= quantiles[p]].value.max()

				if withMean:
					out_q[n,target,p+1] = mean_predictions[n,target]

		return out_q


		






