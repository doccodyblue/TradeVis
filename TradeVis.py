# Dependencies and Configuration
import pandas as pd
import fnmatch
import sys
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import argparse
from colorama import Fore, Style

debug = False


# Configuration Section
# * is working as a wildcard in the comment pattern
# * should also work as a wildcard in the magic number pattern, just give the first few digits of the magic number BUT AS A STRING, followed by *
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
		'comment_patterns': ['Perceptrade*']
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
		'comment_patterns': ['ASEnhancedBO*']
	},
	'MeetAlgo Strategy Builder EA EURGBP': {
		'magic_numbers': [23423497],
		'comment_patterns': ['']
	},
	'AS-LondonBreakout': {
		'magic_numbers': [88378, 88374],
		'comment_patterns': ['AS-LondonBreakout*']
	},
	'BotAGI-FX': {
		'magic_numbers': [100, 101],
		'comment_patterns': ['EA MT5 BotAGI*']
	},
	'ManHedger': {
		'magic_numbers': [3113311],
		'comment_patterns': ['']
	},
	'EA-Studio 99531851 GJ-M15': {
		'magic_numbers': [99531851],
		'comment_patterns': ['99531851']
	},
	'EA-Studio 59796456': {
		'magic_numbers': [59796456],
		'comment_patterns': ['59796456']
	},
	'AS-HoldOverNight': {
		'magic_numbers': [92883],
		'comment_patterns': ['AS-HoldOvernight']
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
		data['Symbol'] = data['Symbol'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)
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
    if magic == 0 and pd.isna(comment):
        if debug:
            print(f"Magic and comment are NaN, returning 'manual'")
        return 'Manual'

    for identifier, details in ALGO_MAPPING_CONFIG.items():
        # Convert magic number to string for matching
        magic_str = str(magic)
        for magic_number in details['magic_numbers']:
            # Convert each magic number in the config to string
            magic_number_str = str(magic_number)
            # Check if the magic number in the config ends with a *
            if magic_number_str.endswith('*'):
                # If it does, remove the * and use startswith for matching
                magic_number_str = magic_number_str[:-1]
                if magic_str.startswith(magic_number_str):
                    if debug: print(f"Matched {magic} to {identifier} using magic numbers")
                    return identifier
            else:
                # If it doesn't, use equality check for matching
                if magic_str == magic_number_str:
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
        first_trade_date=pd.NamedAgg(column='OpenTime', aggfunc='min'),
        last_trade_date=pd.NamedAgg(column='OpenTime', aggfunc='max')  # Added this line
    ).reset_index()

    # Post-aggregation calculations
    grouped['Risk_Reward_Ratio'] = grouped['avg_win_trade'] / abs(grouped['avg_loss_trade'])
    grouped['Profit_Factor'] = grouped['total_profit'] / abs(grouped['total_losses'])
    grouped['Score'] = grouped['total_profit'] - grouped['max_drawdown']

    return grouped

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
                         "<br>Last Trade Made: " + data['last_trade_date'].astype(str) + \
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
	if max_robustness == min_robustness:
		robustness_normalized = {k: 2.5 for k in robustness_mapping}  # Assigning a default middle value
	else:
		robustness_normalized = {k: 1 + 4 * (v - min_robustness) / (max_robustness - min_robustness) for k, v in
								 robustness_mapping.items()}

	#robustness_normalized = {k: 1 + 4 * (v - min_robustness) / (max_robustness - min_robustness) for k, v in
	#						 robustness_mapping.items()}
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

	# Add balance argument
	parser.add_argument("--balance", type=float, default=10000.0, help="Account balance.")

	# Add leverage argument
	parser.add_argument("--leverage", type=int, default=50, help="Account leverage.")

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



def populate_algo_stats_from_dataframe(df):
	algo_stats = {}

	for _, row in df.iterrows():
		algo_name = row['Identifier_Combined']

		algo_stats[algo_name] = {
			'total_profit': row['total_profit'],
			'Risk_Reward_Ratio': row['Risk_Reward_Ratio'],
			'Profit_Factor': row['Profit_Factor'],
			'number_of_trades': row['number_of_trades']
		}

	return algo_stats


class Taxes:

	@staticmethod
	def get_summary(data):
		"""
		Calculate and print the yearly and monthly summaries of profit and loss from the data.

		Args:
			data (pd.DataFrame): The trading data.
		"""
		# Yearly Summary
		data['Year'] = data['OpenTime'].dt.year
		yearly_summaries = data.groupby('Year').apply(lambda x: pd.Series({
			'Total Profit': x['Profit'][x['Profit'] > 0].sum(),
			'Total Loss': x['Profit'][x['Profit'] <= 0].sum()
		})).reset_index()

		print("\n📅 Yearly Summary 📅")
		for index, row in yearly_summaries.iterrows():
			sum_profit_loss = row['Total Profit'] + row['Total Loss']
			if sum_profit_loss < 0:
				sum_profit_loss = f"{Fore.RED}€{sum_profit_loss:.2f}{Style.RESET_ALL}"
			else:
				sum_profit_loss = f"€{sum_profit_loss:.2f}"
			print(f"\n{row['Year']}:")
			print(f"Total Profit: €{row['Total Profit']:.2f}")
			print(f"Total Loss: €{row['Total Loss']:.2f}")
			print(f"Sum: {sum_profit_loss}")
			if row['Total Loss'] <= -20000:
				print("⚠️ Warning: Your yearly loss exceeds €20,000! ⚠️")

		# Monthly Summary
		data['YearMonth'] = data['OpenTime'].dt.strftime('%B %Y')
		monthly_summaries = data.groupby('YearMonth').apply(lambda x: pd.Series({
			'Total Profit': x['Profit'][x['Profit'] > 0].sum(),
			'Total Loss': x['Profit'][x['Profit'] <= 0].sum()
		}))

		# Sort by year and month
		monthly_summaries = monthly_summaries.reset_index()
		monthly_summaries['SortDate'] = pd.to_datetime(monthly_summaries['YearMonth'], format='%B %Y')
		monthly_summaries = monthly_summaries.sort_values(by='SortDate').drop(columns=['SortDate'])

		print("\n📆 Monthly Summary By Year 📆")
		for index, row in monthly_summaries.iterrows():
			sum_profit_loss = row['Total Profit'] + row['Total Loss']
			if sum_profit_loss < 0:
				sum_profit_loss = f"{Fore.RED}€{sum_profit_loss:.2f}{Style.RESET_ALL}"
			else:
				sum_profit_loss = f"€{sum_profit_loss:.2f}"

			print(f"{row['YearMonth']}: 		Sum: {sum_profit_loss}")
			print(f"  Total Profit: €{row['Total Profit']:.2f}")
			print(f"  Total Loss: €{row['Total Loss']:.2f}\n")

	@staticmethod
	def plot_monthly_summary(data):
		"""Plot the monthly summaries of profit and loss using Plotly."""
		# Extracting monthly data
		data['YearMonth'] = data['OpenTime'].dt.strftime('%B %Y')
		monthly_summaries = data.groupby('YearMonth').apply(lambda x: pd.Series({
			'Total Profit': x['Profit'][x['Profit'] > 0].sum(),
			'Total Loss': x['Profit'][x['Profit'] <= 0].sum()
		}))

		# Sort by month and year
		month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
					   'November', 'December']
		monthly_summaries = monthly_summaries.reset_index()
		monthly_summaries['Year'] = pd.to_datetime(monthly_summaries['YearMonth'], format='%B %Y').dt.year
		monthly_summaries['Month'] = pd.Categorical(monthly_summaries['YearMonth'].str.split(' ').str[0],
													categories=month_order, ordered=True)
		monthly_summaries = monthly_summaries.sort_values(by=['Year', 'Month'])

		# Calculate the cumulative sum of profit and loss
		monthly_summaries['Cumulative Sum'] = (
					monthly_summaries['Total Profit'] + monthly_summaries['Total Loss']).cumsum()

		# Plotting
		fig = go.Figure()

		fig.add_trace(go.Bar(
			x=monthly_summaries['YearMonth'],
			y=monthly_summaries['Total Profit'],
			name='Profit',
			marker_color='green'
		))

		fig.add_trace(go.Bar(
			x=monthly_summaries['YearMonth'],
			y=monthly_summaries['Total Loss'],
			name='Loss',
			marker_color='red'
		))

		# Add the cumulative sum line using a scatter trace
		fig.add_trace(go.Scatter(
			x=monthly_summaries['YearMonth'],
			y=monthly_summaries['Cumulative Sum'],
			mode='lines',
			name='Cumulative Sum',
			line=dict(color="blue", width=2, dash="dash")
		))

		fig.update_layout(
			title="Monthly Profit & Loss",
			xaxis_title="Month",
			yaxis_title="Amount (€)",
			barmode='relative',
			plot_bgcolor="black",
			paper_bgcolor="black",
			font=dict(color="white"),
			xaxis=dict(gridcolor="gray", tickangle=-45),  # added tickangle for better readability
			yaxis=dict(gridcolor="gray")
		)

		fig.show()

def setup_cli_args():
	parser = argparse.ArgumentParser(description="Trade Analysis Tool")

	# Add data file path argument
	parser.add_argument("data_file_path", type=str, help="Path to the data file.")

	# Add days back argument
	parser.add_argument("days_back", type=int, help="Number of days back to consider.")

	# Add balance argument
	parser.add_argument("--balance", type=float, default=10000.0, help="Account balance.")

	# Add leverage argument
	parser.add_argument("--leverage", type=int, default=50, help="Account leverage.")

	# Add -d flag for drawdown visualization
	parser.add_argument("-d", "--drawdown", action="store_true",
						help="Enable drawdown visualization. If not set, don't visualize the drawdowns.")

	# Add -x flag for Excel export
	parser.add_argument("-x", "--export", action="store_true",
						help="Enable Excel export. If not set, don't export to Excel.")

	args = parser.parse_args()
	return args

def export_to_excel(data, filename):
	# Extracting monthly data
	data['YearMonth'] = data['OpenTime'].dt.strftime('%B %Y')
	data['Year'] = data['OpenTime'].dt.year
	data['Month'] = data['OpenTime'].dt.strftime('%B')

	# Create a Pandas Excel writer using XlsxWriter as the engine
	writer = pd.ExcelWriter(filename, engine='xlsxwriter')

	# Define cell formats
	header_format = writer.book.add_format({'bold': True, 'font_color': 'white', 'bg_color': 'black', 'border': 1})
	cell_format = writer.book.add_format({'font_color': 'black', 'bg_color': 'white', 'border': 1})

	for year in data['Year'].unique():
		yearly_data = data[data['Year'] == year]
		monthly_summaries = yearly_data.groupby('Month').apply(lambda x: pd.Series({
			'Total Profit': x['Profit'][x['Profit'] > 0].sum(),
			'Total Loss': x['Profit'][x['Profit'] <= 0].sum(),
			'Trading Costs': (x['Commission'] + x['Swap']).sum()
		}))

		# Sort by month
		month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
					   'November', 'December']
		monthly_summaries = monthly_summaries.reset_index()
		monthly_summaries['Month'] = pd.Categorical(monthly_summaries['Month'], categories=month_order, ordered=True)
		monthly_summaries = monthly_summaries.sort_values(by='Month')

		# Add yearly summary
		yearly_summary = pd.DataFrame([monthly_summaries.sum(numeric_only=True)], columns=monthly_summaries.columns, index=['Yearly Summary'])
		monthly_summaries = pd.concat([monthly_summaries, yearly_summary])

		# Write each dataframe to a different worksheet
		monthly_summaries.to_excel(writer, sheet_name=str(year), index=False)

		# Apply formats to the cells
		worksheet = writer.sheets[str(year)]
		for idx, col in enumerate(monthly_summaries):  # loop through all columns
			series = monthly_summaries[col]
			max_len = max((
				series.astype(str).map(len).max(),  # len of largest item
				len(str(series.name))  # len of column name/header
				)) + 1  # adding a little extra space
			worksheet.set_column(idx, idx, max_len, cell_format)  # set column width
			worksheet.write(0, idx, series.name, header_format)  # write header

	# Close the Pandas Excel writer and output the Excel file
	writer.save()

def main(data_file_path, days_back, show_drawdown=False, export=False):
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

	# Assuming `algo_evaluation` is the DataFrame you got from your `evaluate_algorithms` function
	algo_evaluation = evaluate_algorithms(filtered_data)

	# Populate algo_stats
	algo_stats = populate_algo_stats_from_dataframe(algo_evaluation)

	Taxes.get_summary(data)
	Taxes.plot_monthly_summary(data)
	if export:
		account_name = data_file_path.split('/')[-1].replace('.txt', '')
		filename = f'taxes_overview_{account_name}.xlsx'
		export_to_excel(data, filename)

if __name__ == "__main__":
	args = setup_cli_args()
	data_file_path = args.data_file_path
	days_back = args.days_back
	show_drawdown = args.drawdown
	balance = args.balance
	leverage = args.leverage
	export = args.export
	account_info = {'balance': balance, 'leverage': leverage}

	try:
		days_back = int(sys.argv[2])
	except ValueError:
		print("Error: Please provide a valid number for days back.")
		sys.exit(1)
	main(data_file_path, days_back, show_drawdown, export)
