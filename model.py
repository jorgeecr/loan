import argparse
from os import path
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, 
                             auc, 
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, 
                                     GridSearchCV)



def get_features(dir):
	reports = pd.read_csv(dir + "credit_reports.csv")
	users = pd.read_csv(dir + "users.csv")

	index = users.set_index('id').index

	income = users[['id','monthly_income','monthly_outcome']].set_index('id')
	r_numeric = reports[['user_id', 'amount_to_pay_next_payment', 'number_of_payments_due','maximum_credit_amount', 'current_balance', 'credit_limit','past_due_balance','worst_delinquency_past_due_balance']]
	r_numeric = r_numeric.groupby(['user_id']).sum()
	r_numeric = pd.concat([r_numeric, income], axis=1) 

	feats = reports.join(users, on='user_id')

	feat_numericos =r_numeric[['amount_to_pay_next_payment',
	                             'number_of_payments_due', 
	                             'credit_limit', 'past_due_balance', 
	                             'worst_delinquency_past_due_balance', 
	                             'monthly_income', 
	                             'monthly_outcome','current_balance']]
	number_of_loans = reports[['user_id','institution']].groupby(['user_id']).count()
	number_of_loans=number_of_loans.rename(columns={'institution':'number_of_loans'}) 

	loans = feats[['user_id','institution','current_balance']]
	zero_balance = loans[loans['current_balance'] == 0]
	zero_balance = zero_balance.groupby('user_id').count()

	paid_loans = pd.DataFrame(index=index,columns=zero_balance.columns)
	paid_loans.loc[zero_balance.index,zero_balance.columns] = zero_balance
	paid_loans = paid_loans.fillna(0)
	paid_loans = paid_loans.drop('institution',1)
	paid_loans = paid_loans.rename(columns={'current_balance':'number_of_paid_loans'})

	number_of_inst=reports[['user_id','institution']].groupby('user_id')['institution'].nunique()
	number_of_inst=number_of_inst.to_frame()
	number_of_inst=number_of_inst.rename(columns={'institution':'number_of_inst'})

	loans_by_inst = feats[['user_id','institution']]
	bank_loans_index = feats['institution']=='BANCO'
	bank_loans = loans_by_inst[bank_loans_index]
	bank_loans = bank_loans.groupby('user_id').count()

	number_of_bank_loans = pd.DataFrame(index=index,columns=bank_loans.columns)
	number_of_bank_loans.loc[bank_loans.index,bank_loans.columns] = bank_loans
	number_of_bank_loans = number_of_bank_loans.fillna(0)
	number_of_bank_loans = number_of_bank_loans.rename(columns={'institution':'bank_loans'})

	r_inst_balance = feats[['user_id','institution','current_balance']]
	paid_bank_loans_index = feats['current_balance']==0

	paid_bank_loans = r_inst_balance[bank_loans_index & paid_bank_loans_index]
	paid_bank_loans = paid_bank_loans.groupby('user_id').count()

	number_of_paid_bank_loans = pd.DataFrame(index=index,columns=paid_bank_loans.columns)
	number_of_paid_bank_loans.loc[paid_bank_loans.index,paid_bank_loans.columns] = paid_bank_loans
	number_of_paid_bank_loans = number_of_paid_bank_loans.fillna(0)
	number_of_paid_bank_loans = number_of_paid_bank_loans.drop('institution',1)
	number_of_paid_bank_loans = number_of_paid_bank_loans.rename(columns={"current_balance":"paid_bank_loans"})


	frequencies = reports[['user_id','institution','payment_frequency',		#frequency of payments due
	                 'amount_to_pay_next_payment']] 
	frequencies = frequencies[frequencies.amount_to_pay_next_payment !=0]	
	frequencies = frequencies.drop(['amount_to_pay_next_payment'],1)		# we're only considering frequency not amount
	frequencies = frequencies.groupby(['user_id','payment_frequency']).agg(['count'])
	frequencies = frequencies.pivot_table('institution',['user_id'],'payment_frequency')
	frequencies = frequencies.fillna(0)
	frequencies = frequencies['count']
	if 'Catorcenal' in frequencies.columns:
		frequencies = frequencies.drop(['Catorcenal'],1)
	if 'Trimestral' in frequencies.columns:
		frequencies = frequencies.drop(['Trimestral'],1)

	pagos_pend_categoria = pd.DataFrame(index=index,columns=frequencies.columns)
	pagos_pend_categoria.loc[frequencies.index,frequencies.columns] = frequencies
	pagos_pend_categoria = pagos_pend_categoria.fillna(0)

	r_credit_type = feats[['user_id','credit_type']]
	credit_card_index = feats['credit_type']=='Tarjeta de Crédito'
	credit_card_loans = r_credit_type[credit_card_index]
	credit_card_loans = credit_card_loans.groupby('user_id').count()

	numb_credit_cards = pd.DataFrame(index=index,columns=credit_card_loans.columns)		#revolving credit is a key feature
	numb_credit_cards.loc[numb_credit_cards.index,numb_credit_cards.columns] = credit_card_loans
	numb_credit_cards = numb_credit_cards.fillna(0)
	numb_credit_cards = numb_credit_cards.rename(columns={"credit_type":"numb_credit_cards"})

	worst_delinquency = reports[['user_id', 'worst_delinquency']]
	worst_delinquency = worst_delinquency.groupby(['user_id']).agg(['max'])
	worst_delinquency = worst_delinquency.fillna(0)

	feature_names = [feat_numericos,number_of_loans,number_of_inst,numb_credit_cards, worst_delinquency,paid_loans]
	return pd.concat(feature_names, axis=1)

	#if set is labelled
	if 'class' in users.columns:
		y = users['class']
	else:
		y = pd.DataFrame(columns=['class'])
	
	return pd.concat([X,y],axis=1)


def get_labels(dir):
	users = pd.read_csv(dir + "users.csv")
	return users[['id','class']].set_index('id')

def train(data_dir, output_dir):
	X = get_features(data_dir)
	y = get_labels(data_dir)
	
	scaler = StandardScaler()
	scaler.fit(X)
	X_scaled = scaler.transform(X)

	dump(scaler, output_dir + 'scaler.joblib')
	print('Scaler saved.')
	
	X_train, X_test, y_train, y_test = train_test_split(X_scaled,y.values, test_size=0.2)
	pipe = Pipeline([('classifier' , LogisticRegression())])
	param_grid = [
	    {'classifier' : [LogisticRegression()],
	     'classifier__penalty' : ['l1', 'l2'],
	    'classifier__C' : np.logspace(0, 3, 10, 2),
	    'classifier__solver' : ['liblinear']
	    },
	    {'classifier' : [xgb.XGBClassifier()],
	     'classifier__min_child_weight' : [1, 5, 10],
	     'classifier__gamma' : [0.0,0.1,0.5, 1, 1.5,2],
	     'classifier__subsample' : [0.6, 0.8, 1.0],
	     'classifier__colsample_bytree' : [0.6, 0.8, 1.0],
	     'classifier__max_depth' : [2, 3, 4]
	     },
	    {'classifier' : [RandomForestClassifier()],
	    'classifier__n_estimators' : list(range(10,101,10)),
	    'classifier__max_features' : list(range(1,12,3))}
	]

	clf = GridSearchCV(pipe, param_grid = param_grid, scoring='roc_auc', cv = 5, verbose=True, n_jobs=-1)
	print('Training the model...')
	best_clf = clf.fit(X_train, y_train.ravel())
	print('Model trained succesfuly')
	print(f'Best parameters: {best_clf.best_params_}')
	print(f'ROC AUC (Training set): {best_clf.best_score_:.4f}')
	print(f'ROC AUC (Test set): {roc_auc_score(y_test,best_clf.predict(X_test)):.4f}')
	dump(best_clf, output_dir + 'classifier.joblib')
	print('Best model saved succesfuly')


def predict(data_dir, model_dir):
	clf_filename = model_dir + 'classifier.joblib'
	scaler_filename = model_dir + 'scaler.joblib'
	if path.exists(clf_filename):
		scaler = load(scaler_filename)
		clf = load(clf_filename)
		X = get_features(data_dir)
		X_scaled = scaler.transform(X)
		predictions = clf.predict(X_scaled) 	#it's possible to use predict_proba too
		users = pd.read_csv(data_dir + "users.csv")
		users['prediction'] = predictions
		users.to_csv(data_dir + "predictions.csv", index=False)
		print('Predictions file created at ' + data_dir)
	else:
		print('Model not found; you must train the model before making any predictions.')   

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('mode', choices=['train','predict'])
	parser.add_argument('input') #data dir 
	parser.add_argument('-o', '--output') #model dir

	args = parser.parse_args()
	if args.mode == 'train':
		if args.input == ".":
			input_dir = ""
		else:
			input_dir = args.input 
		if args.output:
			if args.output == ".":
				output_dir = ""
			else:
				output_dir = args.output
		else:
			output_dir = ""
		train(input_dir, output_dir)
	elif args.mode == 'predict':
		if args.input == ".":
			input_dir = ""
		else:
			input_dir = args.input 
		if args.output:
			if args.output == ".":
				model_dir = ""
			else:
				model_dir = args.output
		else:
			model_dir = ""
		predict(input_dir, model_dir)


if __name__ == '__main__':
    main()
