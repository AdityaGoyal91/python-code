def save_model_image(model_to_save, model_name, location='gdm_bid_model'):

    if location == 'gdm_bid_model':
        path = 'analytics_scratch/model_images/gdm_bid_model/'
    if location == 'gdm_incremental_model':
        path = 'analytics_scratch/model_images/gdm_incremental_model/'
    if location == 'growth_forecasting_model':
        path = 'analytics_scratch/model_images/growth_forecasting_model/'


    hc = sc._jsc.hadoopConfiguration()

    session = boto3.Session(
        aws_access_key_id=hc.get("fs.s3.awsAccessKeyId"),
        aws_secret_access_key=hc.get("fs.s3.awsSecretAccessKey")
    )

    s3_resource = session.resource('s3')

    model_image = pickle.dumps(model_to_save)

    bucket = 'dwh-poshmark-production'
    key = path + model_name

    model_obj = s3_resource.Object(bucket, key)
    model_obj.put(Body=model_image)

    print("Model stored in S3, please go check now")


def load_model_image(model_name, location='gdm_bid_model'):

    if location == 'gdm_bid_model':
        path = 'analytics_scratch/model_images/gdm_bid_model/'
    if location == 'gdm_incremental_model':
        path = 'analytics_scratch/model_images/gdm_incremental_model/'
    if location == 'growth_forecasting_model':
        path = 'analytics_scratch/model_images/growth_forecasting_model/'


    hc = sc._jsc.hadoopConfiguration()

    session = boto3.Session(
        aws_access_key_id=hc.get("fs.s3.awsAccessKeyId"),
        aws_secret_access_key=hc.get("fs.s3.awsSecretAccessKey")
    )
    s3_resource = session.resource('s3')

    bucket = 'dwh-poshmark-production'
    key = path + model_name

    loaded_model = pickle.loads(s3_resource.Bucket(bucket).Object(key).get()['Body'].read())

    return loaded_model
