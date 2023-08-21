# Dependencies and Configuration
import pandas as pd
import fnmatch
import sys
import plotly.express as px
import plotly.graph_objects as go


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
	'AS-EnhancedBO-DAX40': {
		'magic_numbers': [1491],
		'comment_patterns': ['']
	},
	'AS-EnhancedBO-NQ100': {
		'magic_numbers': [1492],
		'comment_patterns': ['']
	},
	'MeetAlgo Strategy Builder EA EURGBP': {
		'magic_numbers': [23423497],
		'comment_patterns': ['']
	}
}

COLUMN_NAMES = ["TradeID", "Symbol", "Lotsize", "Direction", "OpenPrice", "OpenTime", "ExitPrice", "ExitTime",
                "Commission", "Swap", "Profit", "StopLoss", "TakeProfit", "MagicNumber", "Comment"]

# Loading and preprocessing
def load_and_preprocess_data(filepath):
    try:
        data = pd.read_csv(filepath, sep=";", names=COLUMN_NAMES, index_col=False)
        data['OpenTime'] = pd.to_datetime(data['OpenTime'], format='%Y.%m.%d %H:%M')
        data['ExitTime'] = pd.to_datetime(data['ExitTime'], format='%Y.%m.%d %H:%M')
        data['Symbol'] = data['Symbol'].str.replace('[', '').str.replace(']', '')
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def filter_data_by_date(data, days_back):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days_back)
    return data[(data['OpenTime'] >= start_date) & (data['OpenTime'] <= end_date)]

# Mapping and Evaluating
def map_algo_name(magic, comment):
    for identifier, details in ALGO_MAPPING_CONFIG.items():
        # Check magic numbers first
        if magic in details['magic_numbers']:
            return identifier
        # If not found in magic numbers, then check comment patterns
        for pattern in details['comment_patterns']:
            if fnmatch.fnmatch(str(comment), pattern):
                return identifier
    return comment if comment != 0 else str(magic) if magic != 0 else "UNKNOWN"


def evaluate_algorithms_helper(data, group_by_cols):
    data['Identifier_Combined'] = data.apply(lambda row: map_algo_name(row['MagicNumber'], row['Comment']), axis=1)
    grouped = data.groupby(group_by_cols).agg(
        total_profit=pd.NamedAgg(column='Profit', aggfunc=sum),
        number_of_trades=pd.NamedAgg(column='TradeID', aggfunc='count'),
        max_drawdown=pd.NamedAgg(column='Profit', aggfunc='min'),
        first_trade_date=pd.NamedAgg(column='OpenTime', aggfunc='min'),
        winning_trades=pd.NamedAgg(column='Profit', aggfunc=lambda x: (x > 0).sum()),
        losing_trades=pd.NamedAgg(column='Profit', aggfunc=lambda x: (x <= 0).sum())
    ).reset_index()
    grouped['Score'] = grouped['total_profit'] - grouped['max_drawdown']
    return grouped

def evaluate_algorithms(data):
    grouped = evaluate_algorithms_helper(data, ['Identifier_Combined'])
    # Calculate robustness score
    grouped['Robustness_Score'] = grouped['total_profit'] / abs(grouped['max_drawdown'])
    return grouped

def evaluate_algorithms_by_symbol(data):
    return evaluate_algorithms_helper(data, ['Identifier_Combined', 'Symbol'])

def evaluate_algorithms_cumulative(data):
    data['Identifier_Combined'] = data.apply(lambda row: map_algo_name(row['MagicNumber'], row['Comment']), axis=1)
    data = data.sort_values('OpenTime')
    data['CumulativeProfit'] = data.groupby('Identifier_Combined')['Profit'].cumsum()
    return data


def plot_total_profit_by_symbol(data):
	# Calculate number of days since the first trade for each algorithm
	data['days_since_first_trade'] = (pd.Timestamp.now() - data['first_trade_date']).dt.days

	# Create custom hover text
	data['hover_text'] = "Algorithm: " + data['Identifier_Combined'] + \
						 "<br>Total Profit: " + data['total_profit'].astype(str) + \
						 "<br>Number of Trades: " + data['number_of_trades'].astype(str) + \
						 "<br>Winning Trades: " + data['winning_trades'].astype(str) + \
						 "<br>Losing Trades: " + data['losing_trades'].astype(str) + \
						 "<br>Days Since First Trade: " + data['days_since_first_trade'].astype(str) + \
						 "<br>Traded Symbol: " + data['Symbol']

	fig = px.bar(data,
				 x='total_profit',
				 y='Identifier_Combined',
				 color='Symbol',
				 orientation='h',
				 labels={'Identifier_Combined': 'Algorithm', 'total_profit': 'Total Profit', 'Symbol': 'Traded Symbol'},
				 title='Total Profit by Algorithm and Traded Symbol',
				 hover_data={'total_profit': False, 'Symbol': False, 'Identifier_Combined': False},
				 hover_name='hover_text')

	fig.update_layout(
		showlegend=True,
		plot_bgcolor="black",
		paper_bgcolor="black",
		font=dict(color="white"),
		xaxis=dict(gridcolor="gray"),
		yaxis=dict(gridcolor="gray")
	)
	fig.show()


def plot_cumulative_profit_over_time(data, cumulative_data):
    algo_evaluation = evaluate_algorithms(data)
    robustness_mapping = dict(zip(algo_evaluation['Identifier_Combined'], algo_evaluation['Robustness_Score']))
    max_robustness = max(robustness_mapping.values())
    min_robustness = min(robustness_mapping.values())
    robustness_normalized = {k: 1 + 4 * (v - min_robustness) / (max_robustness - min_robustness) for k, v in
                             robustness_mapping.items()}
    ordered_identifiers = sorted(robustness_mapping, key=robustness_mapping.get, reverse=True)
    fig = go.Figure()
    for identifier in ordered_identifiers:
        df = cumulative_data[cumulative_data['Identifier_Combined'] == identifier]
        linewidth = robustness_normalized.get(identifier, 1)
        fig.add_trace(
            go.Scatter(x=df['OpenTime'], y=df['CumulativeProfit'], mode='lines', name=identifier,
                       line=dict(width=linewidth))
        )
    data_total = data.copy()
    data_total['CumulativeProfit'] = data['Profit'].cumsum()
    fig.add_trace(go.Scatter(x=data_total['OpenTime'], y=data_total['CumulativeProfit'], mode='lines',
                             name='All Algos Combined', line=dict(color='red', width=6, dash='dot')))
    fig.update_layout(
        title='Cumulative Profit Over Time by Algorithm',
        hovermode="x unified",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        xaxis=dict(gridcolor="gray"),
        yaxis=dict(gridcolor="gray")
    )
    fig.show()


# Main Function
def main(data_file_path, days_back):
    data = load_and_preprocess_data(data_file_path)
    filtered_data = filter_data_by_date(data, days_back)
    algo_evaluation = evaluate_algorithms_by_symbol(filtered_data)
    plot_total_profit_by_symbol(algo_evaluation)
    cumulative_data = evaluate_algorithms_cumulative(filtered_data)
    plot_cumulative_profit_over_time(filtered_data, cumulative_data)

# Execution
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
