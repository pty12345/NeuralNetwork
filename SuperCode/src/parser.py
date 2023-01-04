import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='SMD',
                    help="dataset from ['synthetic', 'SMD']")
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='LPC_AD',
                    help="model name")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--unsave', 
					action='store_true', 
					help="unsave the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")
parser.add_argument('--less_ratio', 
					type=float, 
					required=False,
					default=0.4,
                    help="ratio of less training dataset")
args = parser.parse_args()