import math as m
import logging
import ipaddress
import pandas as pd
from scipy.stats import ks_2samp
from connectors.db_connector import RedshiftConnector

logger = logging.getLogger('ip')

def get_ip_data(country: str, pacid=None, query_file: str= 'sql/obtain_ip.sql'):
    conn = RedshiftConnector()
    with open(query_file, 'r') as file:
        query = file.read()
    if pacid == None:
        query = query.format(number=2, country=country, filter_pacid="")
    else:
        filter_pacid = f"and pacid = {pacid}"
        query = query.format(number=2, country=country, filter_pacid=filter_pacid)
    logger.debug(query)
    data = conn.query_df(query)
    return data


def map_networks(ip_data: pd.DataFrame,
                 starting_subnet: int,
                 network_min_views: int,
                 network_starting_id: int,
                 max_subnet: int):
    """
    Given a dataframe with the views and sales per ip. This function return the aggregation
    of these ips based on if the views distribution is similar for two adjacent ips.
    Below the different codes used to define the network (highest_network column)

        highest_network codes:
        0 - non processed (debug purposes)
        100 - highest network
        1 - not enough views to check the network -> check https://sparky.rice.edu//astr360/kstest.pdf
        2 - Not enough statistical significance to reject H0 (subnets belong to same supernet)
        >100 - ip belonging to that supernet will be processed on the next mask loop unless that mask is the MAXIMUM_SUBNET
        """
    higher_subnet = starting_subnet - 1
    df_networks = pd.DataFrame()
    while True:
        current_subnet = starting_subnet - (starting_subnet - (higher_subnet + 1))
        logging.info(f'Working on subnet {current_subnet}')

        if current_subnet == starting_subnet:
            pass
        else:
            ip_data = ip_data.assign(network=ip_data['network']
                                     .apply(lambda x: ipaddress.ip_network(x).supernet(new_prefix=current_subnet)))
            ip_data = ip_data.groupby(['date', 'network'], as_index=False).agg({'sales': 'sum',
                                                                                'views': 'sum',
                                                                                'official_language': 'sum',
                                                                                'non_official_language': 'sum'})
        df_mask = ip_data.groupby(['network'], as_index=False).agg({'sales': 'sum',
                                                                    'views': 'sum',
                                                                    'official_language': 'sum',
                                                                    'non_official_language': 'sum'})
        df_mask = df_mask.assign(parent_network=df_mask['network']
                                 .apply(lambda x: ipaddress.ip_network(x).supernet(new_prefix=higher_subnet)))
        df_mask = df_mask.assign(highest_network=0)  # default non processed code
        parents_networks = df_mask[df_mask['highest_network'] == 0]['parent_network'].unique()

        logging.debug(f'ip_data network example: {ip_data["network"].iloc[0]}')
        logging.debug(f'df_mask network example: {df_mask["network"].iloc[0]}, parent: {df_mask["parent_network"].iloc[0]}')
        logging.debug(f'parent_networks example {parents_networks[0]}')

        for p_network in parents_networks:
            networks = df_mask[(df_mask['parent_network'] == p_network)]['network']
            mask = (df_mask['parent_network'] == p_network)
            if len(networks) == 1:  # There is just one network (no adjacent)
                df_mask.loc[mask, 'highest_network'] = 100

            elif len(networks) == 2:  # There are two adjacent networks
                if df_mask[mask]['views'].sum() < network_min_views:  # Not enough views
                    df_mask.loc[mask, 'highest_network'] = 1
                else:
                    df_daily_views = ip_data[ip_data['network'].isin(networks)].copy()
                    df_daily_views = df_daily_views.assign(date=pd.to_datetime(df_daily_views['date']))
                    first_network = df_daily_views[df_daily_views['network'] == networks.iloc[0]]
                    second_network = df_daily_views[df_daily_views['network'] == networks.iloc[1]]
                    # Create a df with all the daily views distribution for current ip and adjacent ip
                    full_dates_df = pd.date_range(start=pd.Series([first_network['date'].min(),
                                                                   second_network['date'].min()]).min(),
                                                  end=pd.Series([first_network['date'].max(),
                                                                 second_network['date'].max()]).max()
                                                  ).to_frame(name='date')

                    df_daily_views = full_dates_df.merge(first_network, how='left', on='date')
                    df_daily_views = df_daily_views.merge(second_network, how='left', on='date', suffixes=['_first',
                                                                                                           '_second'])
                    del full_dates_df
                    df_daily_views.fillna(0, inplace=True)

                    # Apply statistical significance on views to see whether the two ip can be part of the same network.
                    are_similar, ks_value, min_critical_value = \
                        are_two_distributions_the_same(s1=df_daily_views['views_first'],
                                                       s2=df_daily_views['views_second'])

                    if ks_value < min_critical_value:
                        # Networks belong to same subnet
                        df_mask.loc[((df_mask['network'].isin(networks))
                                     & (df_mask['parent_network'] == p_network))
                        , 'highest_network'] = network_starting_id
                        network_starting_id += 1
                    else:
                        # The network and its adjacent network belong to different subnets
                        df_mask.loc[((df_mask['network'].isin(networks))
                                     & (df_mask['parent_network'] == p_network)), 'highest_network'] = 2

            else:
                raise NotImplementedError(f'We found more than two adjacent networks for parent {p_network}')

        # Add the results of the networks to the analysis dataframe
        if df_networks.empty:
            df_networks = df_mask.copy()
        else:
            df_networks = df_networks.append(df_mask.copy(), ignore_index=True)

        # filter just networks belonging to a higher network and prepare the data to a new network analysis on higher subnetmask
        df_upper_level_networks = df_mask[df_mask['highest_network']>= 100].copy()
        if (current_subnet <= max_subnet) or (df_upper_level_networks.empty):
            if df_upper_level_networks.empty:
                logging.info(f'No networks of the subnet {current_subnet} belonging to a higher subnet: {higher_subnet}')
            break  # Break while
        else:
            df_mask = df_upper_level_networks.assign(network=df_upper_level_networks['parent_network'])
        higher_subnet -= 1
    return df_networks


def are_two_distributions_the_same(s1: pd.Series, s2: pd.Series) -> tuple:
    '''
    Uses kolmogorov-Smirnov test to se whether the two views distribution are the same.
    H0: two distributions are the same
    '''
    ks_value, p_value = ks_2samp(s1.values, s2.values)
    min_critical_value = 1.36 * m.sqrt((len(s1)+len(s2)) / (len(s1)*len(s2))) # https://sparky.rice.edu//astr360/kstest.pdf
    if p_value > 0.05: # Cannot reject null hypothesis H0
        return (True, ks_value, min_critical_value)
    else:  # Reject H0
        return (False, 1, 1)


def language_logic(row: pd.Series, network_min_views: int) -> str:
    if row['views'] >= network_min_views:
        return 'not_enough_views'

    elif (abs(row['official_language'] - row['non_official_language']) >= (min([row['official_language']
                                                                                   ,row['non_official_language']]))):
        if row['official_language'] > row['non_official_language']:
            return 'official'
        elif row['official_language'] < row['non_official_language']:
            return 'non_official'
        else:
            return 'indifferent'
    else:
        return 'error_tagging_language'


def status_logic(row: pd.Series, network_min_views: int, cr_splitter: float) -> str:
    if row['views'] >= network_min_views:
        if row['sales'] == 0:
            return 'not_converting'
        elif row['sales'] / row['views'] >= cr_splitter:
            return 'high_conversion'
        elif (row['sales'] / row['views'] < cr_splitter) and row['sales'] / row['views'] > 0:
            return 'low_conversion'
    else:
        return 'not_enough_views'


def extend_all_subnets_df(df: pd.DataFrame, ip_col='ip', starting_subnet=None):
    if not starting_subnet:
        starting_subnet = int(df[ip_col][1].split('/')[-1])
    df = df.assign(ip_col = df[ip_col].apply(lambda x: '.'.join(x.split('.')[0:-1]) + f'.0/{starting_subnet}'))
    for snet in range(starting_subnet, 0, -1):
        logger.debug(f'Running for subnet {snet}')
        if snet == starting_subnet:
            df[str(snet)] = df[ip_col].apply(lambda x: ipaddress.ip_network(x))
        else:
            df[str(snet)] = df[str(snet + 1)].apply(lambda x: x.supernet(new_prefix=snet))
    return df

def gimli_keep(S : pd.Series):
    dict_min_views = {str(x): 10 * x for x in range(1, 25)}
    min_views = dict_min_views.get(S['IP Network'].split('/')[-1])
    if (S['Views'] >= min_views) & ((S['Sales']/S['Views']) > 0):
        return 1
    else:
        return 0

def new_keep(S: pd.Series):
    dict_min_views = {str(x): 10 * x for x in range (1,25)}
    min_views = dict_min_views.get(S['network'].split('/')[-1])
    if (S['views'] >= min_views) & ((S['sales']/S['views']) > 0):
        return 1
    else:
        return 0

def map_originals_ips(df, df_mapper, col_marked_to_keep: str, df_ip_column: str):
    df_mapper = df_mapper.copy()
    cols = [str(x) for x in range(1, 25, 1)]
    df_mapper[cols] = df_mapper[cols].astype(str)
    df = df[df[col_marked_to_keep]==1]  # Just those networks to keep
    n = len(df[df_ip_column])
    for i, net in enumerate(df[df_ip_column]):
        subnet = net.split('/')[-1]
        df_mapper.loc[df_mapper[df_mapper[subnet] == net].index, col_marked_to_keep] = 1
        print(f'{i} out of {n}')
    return df_mapper[col_marked_to_keep]

