import tempfile
from tqdm import tqdm
import pandas as pd
from tiingo import TiingoClient
import logging
import quantstats as qs
import pandas as pd
import numpy as np
import os

# logger config
logger = logging.getLogger(__name__)
logging.basicConfig(filename='my_log/my.log', encoding='utf-8', level=logging.DEBUG)


# ======================================================================================================================
# section one - pull px data from tiingo api
# ======================================================================================================================

# tiingo config
config = {}
config['session'] = True
config['api_key'] = 'e2f9ec8abc2e37ba5116852a2d847bb70110bd1a'
# initialize
client = TiingoClient(config)

# ticker_gics_industry file - csv to df
tix_df = pd.read_csv('/Users/benjoergens/Desktop/ticker_gics_industry.csv')

# initialize sectors as empty lists
na_sect, hc_sect, re_sect, it_sect, inds_sect, mats_sect, cd_sect, cs_sect, fins_sect, utils_sect, eng_sect, coms_sect \
    = ([] for i in range(12))
# fill sector lists with tix
for ind in tix_df.index:
    if tix_df['Sector'][ind] == 'Health Care':
        hc_sect.append(tix_df['TICKER'][ind])
    elif tix_df['Sector'][ind] == 'Real Estate':
        re_sect.append(tix_df['TICKER'][ind])
    elif tix_df['Sector'][ind] == 'Information Technology':
        it_sect.append(tix_df['TICKER'][ind])
    elif tix_df['Sector'][ind] == 'Industrials':
        inds_sect.append(tix_df['TICKER'][ind])
    elif tix_df['Sector'][ind] == 'Materials':
        mats_sect.append(tix_df['TICKER'][ind])
    elif tix_df['Sector'][ind] == 'Consumer Discretionary':
        cd_sect.append(tix_df['TICKER'][ind])
    elif tix_df['Sector'][ind] == 'Consumer Staples':
        cs_sect.append(tix_df['TICKER'][ind])
    elif tix_df['Sector'][ind] == 'Financials':
        fins_sect.append(tix_df['TICKER'][ind])
    elif tix_df['Sector'][ind] == 'Utilities':
        utils_sect.append(tix_df['TICKER'][ind])
    elif tix_df['Sector'][ind] == 'Energy':
        eng_sect.append(tix_df['TICKER'][ind])
    elif tix_df['Sector'][ind] == 'Communication Services':
        coms_sect.append(tix_df['TICKER'][ind])
    else:
        na_sect.append(tix_df['TICKER'][ind])
logger.info('All tickers sorted by sector')


def exec_section_one(sect_lst, filename):
    # generate 5 yr adjusted price csv files for tix in each sector from tiingo api
    avail_tix = []
    for tix in tqdm(sect_lst, colour='blue', desc='Pulling sector stock price data', position=0, leave=True):
        historical_prices = client.get_ticker_price(tix,
                                                fmt='csv',
                                                startDate='2016-01-02',
                                                endDate='2021-01-02',
                                                frequency='daily')

        # error handling: len 2 if empty list "[]" error; len 26 (after subtracting tix length) if "tix not found" error
        if len(historical_prices) == 2 or (len(historical_prices)-len(tix)) == 26:
            logger.warning(f'No historical px data available for the ticker: {tix}')
        else:
            avail_tix.append(tix)

    ticker_history = client.get_dataframe(avail_tix,
                                          frequency='daily',
                                          fmt='csv',
                                          metric_name='adjClose',
                                          startDate='2016-01-02',
                                          endDate='2021-01-02')

    # generate CSV to save record and make future data handling easier
    ticker_history.to_csv(filename + '.csv')
    logger.info(f'Successfully generated historical px data file for the sector: {filename}')


# ======================================================================================================================
# section two - generate csv file with all sector and portfolio returns
# ======================================================================================================================

# define function to prepare returns dfs from sector px dfs
def prepare_df(og_sect_df, sect_indx):
    # clean and prepare for return calculation
    df = og_sect_df.dropna(thresh=22, axis=1).ffill().fillna(0)
    df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date_formatted'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df.set_index('date_formatted', inplace=True)
    df.drop(columns='datetime', inplace=True)
    # drop penny stocks (criteria = px < $5 for more than a month of returns)
    mask = df.apply(lambda col: (col < 5).sum() > 22)
    filtered_df = df.loc[:, ~mask]

    # calculate returns and adjust first day of new month
    filtered_df = filtered_df.pct_change().fillna(0)
    filtered_df['month'] = filtered_df.index.str.slice(5, 7).astype(int)
    first_of_month = filtered_df['month'].ne(filtered_df['month'].shift(1))
    filtered_df.loc[first_of_month, filtered_df.columns[:-1]] = 0

    # add a 'unique_mo_id' column to allow month categorization
    filtered_df['unique_mo_id'] = pd.factorize(filtered_df.index.str.slice(0, 7))[0]

    # clean up and set up for further processing
    filtered_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    filtered_df.fillna(0, inplace=True)
    filtered_df += 1  # Adjust values for compounding
    filtered_df = filtered_df.drop('month', axis=1)

    logger.info(f'Successfully cleaned historical px data for sector index: {sect_indx}')
    return filtered_df, sect_indx


# define return multiplier void function for return generation
def multiply_sect_returns(factor, df):
    df.iloc[:, :] = factor * df.iloc[:, :].cumprod()


# define sector summer function for sector return generation
def sum_sect_returns(df):
    return df.iloc[:, 1:].sum(axis=1)


# define function to generate each month of returns and calculate sect end vals
def generate_mo_rets(sect_val, sect_rets_df, sect_indx, month):
    raw_mo_sect_df = sect_rets_df[sect_rets_df['unique_mo_id'] == month].copy()
    cln_mo_sect_df = raw_mo_sect_df.loc[:, (raw_mo_sect_df != 1).any(axis=0)]
    if 'unique_mo_id' not in cln_mo_sect_df.columns:
        cln_mo_sect_df['unique_mo_id'] = raw_mo_sect_df['unique_mo_id'].tolist()
    cln_mo_sect_df = cln_mo_sect_df.drop('unique_mo_id', axis=1)
    tix_count = cln_mo_sect_df.shape[1] - 1
    tix_wgts = 1 / tix_count
    tix_vals = sect_val * tix_wgts
    # print('pre mult', cln_mo_sect_df.to_string())
    multiply_sect_returns(tix_vals, cln_mo_sect_df)
    # print('post_mult', cln_mo_sect_df.to_string())
    sect_ret_vals = sum_sect_returns(cln_mo_sect_df)
    return cln_mo_sect_df, sect_ret_vals, sect_indx


# add sector returns to por list for eventual portfolio return generation
def add_to_por_lst(sect_ret_vals, sect_indx, por_lst):
    por_lst[sect_indx - 1].extend(sect_ret_vals)

# compile the sector and portfolio return csv
def compile_por_csv(por_lst, datetimes_lst):
    por_df = pd.DataFrame({'Date': datetimes_lst, 'Indx': por_lst[0], 'HC': por_lst[1], 'RE': por_lst[2],
                           'IT': por_lst[3], 'Ind': por_lst[4], 'Mat': por_lst[5],
                           'CD': por_lst[6], 'CS': por_lst[7], 'Fin': por_lst[8],
                           'Util': por_lst[9], 'Eng': por_lst[10], 'Com': por_lst[11]})
    por_df.set_index('Date', inplace=True)
    # find por_rets by summing across all rows and save in last col
    por_df['Portfolio'] = por_df.sum(axis=1, numeric_only=True)
    por_df.to_csv('sect_por_rets.csv')

# execute the bulk of re-balancing logic
def exec_section_two():
    # once all 12 sector csv's are downloaded, import as dfs - CHANGE FILEPATHS (NOT FOLDER OR FILE NAMES) HERE -
    sector_files = [
        "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/na_sect_pxs.csv",
        "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/hc_sect_pxs.csv",
        "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/re_sect_pxs.csv",
        "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/it_sect_pxs.csv",
        "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/inds_sect_pxs.csv",
        "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/mats_sect_pxs.csv",
        "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/cd_sect_pxs.csv",
        "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/cs_sect_pxs.csv",
        "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/fins_sect_pxs.csv",
        "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/utils_sect_pxs.csv",
        "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/eng_sect_pxs.csv",
        "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/coms_sect_pxs.csv"
    ]

    sector_dct = {}
    for idx, file in enumerate(sector_files, start=1):
        sector_dct[idx] = pd.read_csv(file)

    # initialize portfolio val and sector weights
    og_sect_val = 50000

    # initialize sect lsts
    por_lst = [[] for _ in range(12)]

    updated_sect_vals = 0
    # add all sects to por_lst with monthly re-balancing
    for month_num in tqdm(range(1, 61), colour='blue', desc='Working through monthly data', position=0, leave=True):
        for sect_index in range(1, 13):
            if sect_index == 12:  # update sect vals once there are new vals for each sect
                new_sect_val_sum = 0
                if month_num == 1:  # use og_sect_val
                    clean_df, sect_indx = prepare_df(sector_dct.get(sect_index), sect_index)
                    clean_sect_df, sect_ret_vals, sect_ind = generate_mo_rets(og_sect_val, clean_df, sect_indx,
                                                                              month_num)
                    add_to_por_lst(sect_ret_vals, sect_ind, por_lst)
                    # update sect vals here
                    for sect_returns in por_lst:
                        new_sect_val_sum += sect_returns[-1]
                else:  # use updated_sect_val
                    clean_df, sect_indx = prepare_df(sector_dct.get(sect_index), sect_index)
                    clean_sect_df, sect_ret_vals, sect_ind = \
                        (generate_mo_rets(updated_sect_vals, clean_df, sect_indx, month_num))
                    add_to_por_lst(sect_ret_vals, sect_ind, por_lst)
                    # update sect vals here
                    for sect_returns in por_lst:
                        new_sect_val_sum += sect_returns[-1]
                updated_sect_vals = (new_sect_val_sum / 12)
                logger.info('Successfully compiled monthly returns and vals for all sectors.')
            else:
                if month_num == 1:  # use og_sect_val
                    clean_df, sect_indx = prepare_df(sector_dct.get(sect_index), sect_index)
                    clean_sect_df, sect_ret_vals, sect_ind = generate_mo_rets(og_sect_val, clean_df, sect_indx,
                                                                              month_num)
                    add_to_por_lst(sect_ret_vals, sect_ind, por_lst)
                else:  # use updated_sect_val
                    clean_df, sect_indx = prepare_df(sector_dct.get(sect_index), sect_index)
                    clean_sect_df, sect_ret_vals, sect_ind = \
                        (generate_mo_rets(updated_sect_vals, clean_df, sect_indx, month_num))
                    add_to_por_lst(sect_ret_vals, sect_ind, por_lst)
                logger.info('Compiling data for all sector monthly returns and vals...')
        logger.info('Completed data compilation for one month of returns...')
    logger.info('Completed data compilation for returns.')

    # get datetimes col using arbitrary sect_df
    df, sect_name = prepare_df(list(sector_dct.values())[0], 1)
    df.reset_index(inplace=True)
    datetimes_lst = df['date_formatted'].tolist()
    # round por list to cents
    rounded_por_lst = [[round(value, 2) for value in row] for row in por_lst]

    # call to compile csv
    compile_por_csv(rounded_por_lst, datetimes_lst)

# ======================================================================================================================
# section three - plot returns
# ======================================================================================================================

# execute main html file generation and plotting with quantstats
def exec_section_three():
    # read portfolio returns from csv file
    por_df = pd.read_csv('/sect_pxs/sect_por_rets.csv')
    por_df['Date'] = pd.to_datetime(por_df['Date'])
    por_df.set_index('Date', inplace=True)

    # retrieve historical data for SPY using quantstats
    benchmark = qs.utils.download_returns('SPY')

    # align dates of portfolio returns and benchmark returns
    common_dates = por_df.index.intersection(benchmark.index)

    # set returns here, use sector columm names from the sect_por_rets.csv file to plot sectors instead of portfolio
    returns = por_df.loc[common_dates, 'Portfolio']
    benchmark = benchmark.loc[common_dates]
    benchmark = benchmark.to_frame(name='SPY')

    # generate html report content
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        qs.reports.html(returns, benchmark=benchmark, output=temp_file.name, title='Strategy vs. SPY')
    with open(temp_file.name, 'r') as f:
        html_content = f.read()
    desktop_directory = os.path.join(os.path.expanduser('~'), 'Desktop')

    # write html content to file on desktop if it's not empty
    if html_content:
        report_filename = 'portfolio_report.html'
        # - CHANGE FILEPATH HERE (IF DESIRED) -
        report_filepath = os.path.join(desktop_directory, report_filename)
        try:
            with open(report_filepath, 'w') as f:
                f.write(html_content)
        except Exception as e:
            print(f"Error writing HTML content to file: {e}")
        else:
            print(f"HTML report saved to desktop: {report_filepath}")
    else:
        print("HTML content is empty. Cannot write to file.")


# ======================================================================================================================
# section four - generate finalized returns csv
# ======================================================================================================================

# generate a finalized ticker return csv
def exec_section_four():
    clean_tix_df = tix_df.copy()

    # create a dict mapping industries to their respective csv file paths for easy access
    # - CHANGE FILEPATHS (NOT FOLDER OR FILE NAMES) HERE -
    sector_files = {
        np.nan: "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/na_sect_pxs.csv",
        'Health Care': "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/hc_sect_pxs.csv",
        'Real Estate': "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/re_sect_pxs.csv",
        'Information Technology': "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/it_sect_pxs.csv",
        'Industrials': "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/inds_sect_pxs.csv",
        'Materials': "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/mats_sect_pxs.csv",
        'Consumer Discretionary': "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/cd_sect_pxs.csv",
        'Consumer Staples': "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/cs_sect_pxs.csv",
        'Financials': "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/fins_sect_pxs.csv",
        'Utilities': "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/utils_sect_pxs.csv",
        'Energy': "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/eng_sect_pxs.csv",
        'Communication Services': "/Users/benjoergens/PycharmProjects/Assignment/sect_pxs/coms_sect_pxs.csv"
    }

    # read all csv files into memory
    sector_data = {}
    for sector, file_path in sector_files.items():
        if os.path.exists(file_path):
            sector_data[sector] = pd.read_csv(file_path)

    tix_og_adjpx = []
    tix_last_adjpx = []

    for index, row in tqdm(clean_tix_df.iterrows(), colour='blue', desc='Working through monthly data', position=0, leave=True):
        sector = row['Sector']
        ticker = row['TICKER']
        df = sector_data.get(sector)

        try:
            if df is not None and ticker in df.columns:
                ticker_prices = df[ticker]
                non_zero_prices = ticker_prices[ticker_prices.astype(str).str.replace('.', '').str.isnumeric()].astype(float)
                if not non_zero_prices.empty:
                    tix_og_adjpx.append(non_zero_prices.iloc[0])
                    tix_last_adjpx.append(non_zero_prices.iloc[-1])
                else:
                    tix_og_adjpx.append(np.nan)
                    tix_last_adjpx.append(np.nan)
            else:
                tix_og_adjpx.append(np.nan)
                tix_last_adjpx.append(np.nan)
        except Exception as e:
            logger.error(f"Error processing ticker {ticker} for sector {sector}: {e}")
            tix_og_adjpx.append(np.nan)
            tix_last_adjpx.append(np.nan)

    clean_tix_df['tix_og_adjpx'] = tix_og_adjpx
    clean_tix_df['tix_last_adjpx'] = tix_last_adjpx
    clean_tix_df['cumul_ret'] = ((clean_tix_df['tix_last_adjpx'] / clean_tix_df['tix_og_adjpx']) - 1)
    clean_tix_df['tix_og_adjpx'].round(2)
    clean_tix_df['tix_last_adjpx'].round(2)
    clean_tix_df['cumul_ret'].round(2)
    clean_tix_df['cumul_ret'] = clean_tix_df['cumul_ret'].map('{:.2%}'.format)
    clean_tix_df.fillna('N/A', inplace=True)

    # generate cumulative returns file (step 5) here
    clean_tix_df.to_csv('cumul_rets.csv')


if __name__ == "__main__":
    # section one params: (sector list <lst>, sector px data filename <str>)
    # notes: 
    # - generate sector csvs once at a time in order to avoid potential api overuse
    exec_section_one(inds_sect, 'inds_sect_pxs')
    # section two params: void
    exec_section_two()
    # section three params:
    exec_section_three()
    # section four:
    exec_section_four()
