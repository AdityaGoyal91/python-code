from pmlooker import looker

def get_look_id_sql(look_id, initialize_type = 'config', limit='100'):
    looker_client = looker.lookerAPIClient(initialize_type = initialize_type)
    looker_client.authorize()
    orders_look = looker_client.getlook(look_id=look_id)
    query_id = orders_look['query_id']
    return looker_client.getLookSQL(query_id, limit)
