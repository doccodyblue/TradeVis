import pandas as pd
import fnmatch
import sys
import plotly.express as px

# Configuration Section
# * is working as a wildcard in the comment pattern
ALGO_MAPPING_CONFIG = {
	'EA Zone USDJPY': {
		'magic_numbers': [],
		'comment_patterns': ['EA Zone USDJPY']
	},
	'AS-BreakOut': {
		'magic_numbers': [225846, 22648400, 22648401],
		'comment_patterns': ['AS-BreakOut*']
	},
	'TrendTracer': {
		'magic_numbers': [111],
		'comment_patterns': ['TrendTracer']
	},
	'Perceptrader': {
		'magic_numbers': [12648400, 12648401],
		'comment_patterns': ['Perceptrader*']
	},
	'MA_SB_AV1_SH1': {
		'magic_numbers': [23423450, 23423410],
		'comment_patterns': ['MA_SB_AV1_SH1', 'GRID-MA_SB_AV1_SH1']
	},
	'MeetAlgo Strategy Builder EA': {
		'magic_numbers': [],
		'comment_patterns': ['MeetAlgo Strategy Builder EA']
	},
	'GRID-Ma assist BO': {
		'magic_numbers': [],
		'comment_patterns': ['GRID-Ma assist BO']
	},
	'Waka': {
		'magic_numbers': [],
		'comment_patterns': ['Waka*']
	},
	'AS-EnhancedBO': {
		'magic_numbers': [1490],
		'comment_patterns': ['']
	},
	'AS-EnhancedBO-DAX40_M5': {
		'magic_numbers': [1491],
		'comment_patterns': ['']
	},
	'AS-EnhancedBO-NQ100_M1': {
		'magic_numbers': [1492],
		'comment_patterns': ['']
	}
}

COLUMN_NAMES = ["TradeID", "Symbol", "Lotsize", "Direction", "OpenPrice", "OpenTime", "ExitPrice", "ExitTime", "Commission", "Swap", "Profit", "StopLoss", "TakeProfit", "MagicNumber", "Comment"]

def load_and_preprocess_data(filepath):
    try:
        data = pd.read_csv(filepath, sep=";", names=COLUMN_NAMES, index_col=False)
        data['OpenTime'] = pd.to_datetime(data['OpenTime'], format='%Y.%m.%d %H:%M')
        data['ExitTime'] = pd.to_datetime(data['ExitTime'], format='%Y.%m.%d %H:%M')
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def filter_data_by_date(data, days_back):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days_back)
    return data[(data['OpenTime'] >= start_date) & (data['OpenTime'] <= end_date)]

def map_algo_name(magic, comment):
    for identifier, details in ALGO_MAPPING_CONFIG.items():
        if magic in details['magic_numbers']:
            return identifier
        for pattern in details['comment_patterns']:
            if fnmatch.fnmatch(str(comment), pattern):
                return identifier
    return comment if comment != 0 else str(magic) if magic != 0 else "UNKNOWN"

def evaluate_algorithms(data):
    data['Identifier_Combined'] = data.apply(lambda row: map_algo_name(row['MagicNumber'], row['Comment']), axis=1)
    grouped = data.groupby('Identifier_Combined').agg(
        total_profit=pd.NamedAgg(column='Profit', aggfunc=sum),
        number_of_trades=pd.NamedAgg(column='TradeID', aggfunc='count'),
        max_drawdown=pd.NamedAgg(column='Profit', aggfunc='min')
    ).reset_index()
    grouped['Score'] = grouped['total_profit'] - grouped['max_drawdown']
    return grouped

def plot_total_profit(data):
    fig = px.bar(data, x='total_profit', y='Identifier_Combined', orientation='h',
                 labels={'Identifier_Combined': 'Algorithm', 'total_profit': 'Total Profit'},
                 title='Total Profit by Algorithm')
    fig.update_layout(showlegend=False)
    fig.show()

def plot_cumulative_profit_over_time(data):
    data['CumulativeProfit'] = data.groupby('Identifier_Combined')['Profit'].cumsum()
    data_total = data.copy()
    data_total['Identifier_Combined'] = 'All Algos Combined'
    data_total['CumulativeProfit'] = data['Profit'].cumsum()

    fig = px.line(data, x='OpenTime', y='CumulativeProfit', color='Identifier_Combined',
                  title=f'Cumulative Profit Over Time by Algorithm')
    fig.add_scatter(x=data_total['OpenTime'], y=data_total['CumulativeProfit'], mode='lines',
                    name='All Algos Combined', line=dict(color='red', width=4))
    fig.update_layout(hovermode="x unified")
    fig.show()

def main(data_file_path, days_back):
    data = load_and_preprocess_data(data_file_path)
    filtered_data = filter_data_by_date(data, days_back)
    algo_evaluation = evaluate_algorithms(filtered_data)
    plot_total_profit(algo_evaluation)
    plot_cumulative_profit_over_time(filtered_data)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: script_name.py <data_file_path> <days_back>")
        sys.exit(1)
    data_file_path = sys.argv[1]
    try:
        days_back = int(sys.argv[2])
    except ValueError:
        print("Error: Please provide a valid number for days back.")
        sys.exit(1)
    main(data_file_path, days_back)
