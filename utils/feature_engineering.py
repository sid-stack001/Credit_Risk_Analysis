def create_new_features(data):
    data['income_to_loan_ratio'] = data['income'] / data['loan_amount']
    return data
