# Dependencies and Configuration
import pandas as pd
import fnmatch
import sys
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import argparse

debug = False


# Configuration Section
# * is working as a wildcard in the comment pattern
ALGO_MAPPING_CONFIG = {
	'EA Zone USDJPY': {
		'magic_numbers': [],
		'comment_patterns': ['EA Zone USDJPY']
	},
	'AS-BreakOut': {
		'magic_numbers': [225846],
		'comment_patterns': ['AS-BreakOut*']
	},
	'TrendTracer': {
		'magic_numbers': [111],
		'comment_patterns': ['TrendTracer']
	},
	'Perceptrader': {
		'magic_numbers': [12648400, 12648401, 22648400, 22648401],
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

# filter out symbols from the risk plot
# #NVDA is a share, #* is a wildcard for all shares (e.g. #AAPL, #TSLA)
IGNORE_FILTER = ["#NVDA", "#*"]


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

# Algorithm mapping
def map_algo_name(magic, comment):
	for identifier, details in ALGO_MAPPING_CONFIG.items():
		if magic in details['magic_numbers']:
			if debug: print(f"Matched {magic} to {identifier} using magic numbers")
			return identifier
		for pattern in details['comment_patterns']:
			if fnmatch.fnmatch(str(comment), pattern):
				if debug: print(f"Matched {comment} to {identifier} using comment patterns")
				return identifier
	return comment if comment != 0 else str(magic) if magic != 0 else "UNKNOWN"


def evaluate_algorithms_helper(data, group_by_cols):
	data['Identifier_Combined'] = data.apply(lambda row: map_algo_name(row['MagicNumber'], row['Comment']), axis=1)

	# Aggregation
	grouped = data.groupby(group_by_cols).agg(
		total_profit=pd.NamedAgg(column='Profit', aggfunc=sum),
		total_losses=pd.NamedAgg(column='Profit', aggfunc=lambda x: x[x < 0].sum()),
		avg_win_trade=pd.NamedAgg(column='Profit', aggfunc=lambda x: x[x > 0].mean()),
		avg_loss_trade=pd.NamedAgg(column='Profit', aggfunc=lambda x: x[x <= 0].mean()),
		number_of_trades=pd.NamedAgg(column='TradeID', aggfunc='count'),
		winning_trades=pd.NamedAgg(column='Profit', aggfunc=lambda x: (x > 0).sum()),
		losing_trades=pd.NamedAgg(column='Profit', aggfunc=lambda x: (x <= 0).sum()),
		max_drawdown=pd.NamedAgg(column='Profit', aggfunc='min'),
		first_trade_date=pd.NamedAgg(column='OpenTime', aggfunc='min')
	).reset_index()

	# Post-aggregation calculations
	grouped['Risk_Reward_Ratio'] = grouped['avg_win_trade'] / abs(grouped['avg_loss_trade'])
	grouped['Profit_Factor'] = grouped['total_profit'] / abs(grouped['total_losses'])
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


def calculate_drawdown(cumulative_returns):
	"""
	Calculate the maximum drawdown and its duration.
	"""
	# Calculate the running max
	running_max = np.maximum.accumulate(cumulative_returns)
	# Calculate the drawdown
	drawdown = (cumulative_returns - running_max) / running_max

	# If there's no drawdown, return None for the start and end points
	if all(drawdown == 0):
		return None, None, 0

	# Identify the start and end of the maximum drawdown
	end_point = np.argmin(drawdown)
	if end_point == 0:
		return None, None, 0
	start_point = np.argmax(cumulative_returns[:end_point])

	return start_point, end_point, drawdown[end_point]


def plot_total_profit_by_symbol(data):
	# Calculate number of days since the first trade for each algorithm
	data['days_since_first_trade'] = (pd.Timestamp.now() - data['first_trade_date']).dt.days

	# Create custom hover text
	data['hover_text'] = "Algorithm: " + data['Identifier_Combined'] + \
						 "<br>Total Profit: " + data['total_profit'].astype(str) + \
						 "<br>Number of Trades: " + data['number_of_trades'].astype(str) + \
						 "<br>Winning Trades: " + data['winning_trades'].astype(str) + \
						 "<br>Losing Trades: " + data['losing_trades'].astype(str) + \
						 "<br>Risk-Reward Ratio: " + data['Risk_Reward_Ratio'].round(2).astype(str) + \
						 "<br>Profit Factor: " + data['Profit_Factor'].round(2).astype(str) + \
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


def plot_cumulative_with_drawdown_debug(data, cumulative_data):
	"""
	Plot cumulative profit over time, highlight the maximum drawdown, and print debug information.
	"""
	fig = go.Figure()

	# Calculate and plot for each algorithm
	for algo in cumulative_data['Identifier_Combined'].unique():
		algo_data = cumulative_data[cumulative_data['Identifier_Combined'] == algo]

		# Calculate drawdown information for the algorithm
		start_idx, end_idx, max_drawdown_val = calculate_drawdown(algo_data['CumulativeProfit'].values)

		# Plot cumulative profit
		fig.add_trace(
			go.Scatter(x=algo_data['OpenTime'], y=algo_data['CumulativeProfit'], mode='lines', name=algo,
					   line=dict(width=2))
		)

		# Highlight the maximum drawdown period if there's a valid drawdown
		if start_idx is not None and end_idx is not None:
			start_date = algo_data.iloc[start_idx]['OpenTime']
			end_date = algo_data.iloc[end_idx]['OpenTime']
			y0_value = algo_data.iloc[start_idx]['CumulativeProfit']
			y1_value = algo_data.iloc[end_idx]['CumulativeProfit']

			# Print debug information
			if debug:
				print(f"Algorithm: {algo}")
				print(f"Start Index: {start_idx}, End Index: {end_idx}")
				print(f"Start Date: {start_date}, End Date: {end_date}")
				print(f"Start Value: {y0_value}, End Value: {y1_value}\n")

			fig.add_shape(
				type="rect",
				xref="x",
				yref="y",
				x0=start_date,
				x1=end_date,
				y0=y0_value,
				y1=y1_value,
				fillcolor="red",
				opacity=0.2,
				layer="below",
				line_width=0
			)

	fig.update_layout(
		title='Cumulative Profit Over Time with Maximum Drawdown Highlighted',
		hovermode="x unified",
		plot_bgcolor="black",
		paper_bgcolor="black",
		font=dict(color="white"),
		xaxis=dict(gridcolor="gray"),
		yaxis=dict(gridcolor="gray"),
		shapes=[
			dict(
				type="line",
				xref="paper",
				x0=0,
				x1=1,
				yref="y",
				y0=0,
				y1=0,
				line=dict(color="white", width=2, dash="dash")
			)
		]
	)
	fig.show()


def plot_cumulative_with_drawdown_lines(data, cumulative_data):
    """
    Plot cumulative profit over time with drawdown highlighting and ensure "All Algos Combined" is displayed.
    """
    fig = go.Figure()

    # Plotting cumulative profit for each algorithm
    for identifier in cumulative_data['Identifier_Combined'].unique():
        algo_data = cumulative_data[cumulative_data['Identifier_Combined'] == identifier]
        fig.add_trace(
            go.Scatter(x=algo_data['OpenTime'], y=algo_data['CumulativeProfit'], mode='lines', name=identifier,
                       line=dict(width=2))
        )
        # Highlighting the drawdown for each algorithm
        start_point, end_point, max_drawdown_val = calculate_drawdown(algo_data['CumulativeProfit'].values)
        if start_point is not None and end_point is not None:
            fig.add_shape(
                type="rect",
                x0=algo_data.iloc[start_point]['OpenTime'],
                x1=algo_data.iloc[end_point]['OpenTime'],
                y0=max_drawdown_val,
                y1=algo_data.iloc[start_point]['CumulativeProfit'],
                fillcolor="lightpink",
                opacity=0.5,
                line=dict(color="lightpink", width=0.5)
            )

    # Plotting cumulative profit for "All Algos Combined"
    data_total = data.copy()
    data_total['CumulativeProfit'] = data['Profit'].cumsum()
    fig.add_trace(go.Scatter(x=data_total['OpenTime'], y=data_total['CumulativeProfit'], mode='lines',
                             name='All Algos Combined', line=dict(color='red', width=3, dash='dot')))

    fig.update_layout(
        title='Cumulative Profit Over Time by Algorithm with Drawdown Highlighting',
        hovermode="x unified",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        xaxis=dict(gridcolor="gray"),
        yaxis=dict(gridcolor="gray")
    )
    fig.show()


def plot_cumulative_profit_and_costs(data):
	"""
	Plot cumulative profit and cumulative trading costs/swap over time.
	"""
	fig = go.Figure()

	# Cumulative profit for "All Algos Combined"
	data['CumulativeProfit'] = data['Profit'].cumsum()
	fig.add_trace(
		go.Scatter(x=data['OpenTime'], y=data['CumulativeProfit'], mode='lines', name='Cumulative Profit',
				   line=dict(color='red', width=3))
	)

	# Cumulative trading costs/swap
	data['CumulativeCosts'] = (data['Commission'] + data['Swap']).cumsum()
	fig.add_trace(
		go.Scatter(x=data['OpenTime'], y=data['CumulativeCosts'], mode='lines', name='Cumulative Trading Costs/Swap',
				   line=dict(color='blue', width=3, dash='dot'))
	)

	fig.update_layout(
		title='Cumulative Profit and Trading Costs/Swap Over Time',
		hovermode="x unified",
		plot_bgcolor="black",
		paper_bgcolor="black",
		font=dict(color="white"),
		xaxis=dict(gridcolor="gray"),
		yaxis=dict(gridcolor="gray", title="Amount")
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


def setup_cli_args():
	parser = argparse.ArgumentParser(description="Trade Analysis Tool")

	# Add data file path argument
	parser.add_argument("data_file_path", type=str, help="Path to the data file.")

	# Add days back argument
	parser.add_argument("days_back", type=int, help="Number of days back to consider.")

	# Add -d flag for drawdown visualization
	parser.add_argument("-d", "--drawdown", action="store_true",
						help="Enable drawdown visualization. If not set, don't visualize the drawdowns.")

	args = parser.parse_args()
	return args


def plot_relative_risk_with_algo(data):
	"""
	Plot the lot size (as an indication of relative risk) for each trade over time.
	Display the algorithm name on hover.
	"""
	fig = go.Figure()

	# Filter out symbols that match any pattern in the ignore filter
	for pattern in IGNORE_FILTER:
		data = data[~data['Symbol'].apply(lambda x: fnmatch.fnmatch(x, pattern))]

	# Add hover text to display the algorithm identifier
	data['hover_text'] = data.apply(
		lambda row: f"Algorithm: {row['Identifier_Combined']}<br>Lot Size: {row['Lotsize']}", axis=1)

	# Add a new column to indicate whether the symbol is a share
	data['is_share'] = data['Symbol'].apply(lambda x: x.startswith('#'))

	# Separate the data into shares and other symbols
	shares_data = data[data['is_share']]
	other_data = data[~data['is_share']]

	# Plot the shares with one style
	fig.add_trace(
		go.Scatter(x=shares_data['OpenTime'], y=shares_data['Lotsize'], mode='lines+markers', name='Shares',
				   line=dict(color='red', width=2))
	)

	# Plot the other symbols with a different style
	fig.add_trace(
		go.Scatter(x=other_data['OpenTime'], y=other_data['Lotsize'], mode='lines+markers', name='Other Symbols',
				   line=dict(color='blue', width=2))
	)

	# Highlight trades with lot sizes above a certain threshold
	threshold = data['Lotsize'].mean() + data['Lotsize'].std()
	high_risk_trades = data[data['Lotsize'] > threshold]
	fig.add_trace(
		go.Scatter(x=high_risk_trades['OpenTime'], y=high_risk_trades['Lotsize'], mode='markers',
				   name='High Risk Trades',
				   marker=dict(color='red', size=10), hoverinfo="text", hovertext=high_risk_trades['hover_text'])
	)

	fig.update_layout(
		title='Relative Risk Based on Lot Size Over Time',
		hovermode="closest",
		plot_bgcolor="black",
		paper_bgcolor="black",
		font=dict(color="white"),
		xaxis=dict(gridcolor="gray"),
		yaxis=dict(gridcolor="gray", title="Lot Size"),
		shapes=[
			dict(
				type="line",
				xref="paper",
				x0=0,
				x1=1,
				yref="y",
				y0=threshold,
				y1=threshold,
				line=dict(color="yellow", width=2, dash="dash")
			)
		]
	)
	fig.show()


# Main Function
def main(data_file_path, days_back, show_drawdown=False):
	data = load_and_preprocess_data(data_file_path)
	filtered_data = filter_data_by_date(data, days_back)
	algo_evaluation = evaluate_algorithms_by_symbol(filtered_data)
	plot_total_profit_by_symbol(algo_evaluation)
	cumulative_data = evaluate_algorithms_cumulative(filtered_data)
	if show_drawdown:
		plot_cumulative_with_drawdown_lines(filtered_data, cumulative_data)
	else:
		plot_cumulative_profit_over_time(filtered_data, cumulative_data)
	plot_relative_risk_with_algo(filtered_data)
	plot_cumulative_profit_and_costs(filtered_data)

# Execution
if __name__ == "__main__":
	args = setup_cli_args()
	data_file_path = args.data_file_path
	days_back = args.days_back
	show_drawdown = args.drawdown
	try:
		days_back = int(sys.argv[2])
	except ValueError:
		print("Error: Please provide a valid number for days back.")
		sys.exit(1)
	main(data_file_path, days_back, show_drawdown)
