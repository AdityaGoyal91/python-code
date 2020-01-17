def load_model_image(model_name, location):

    hc = sc._jsc.hadoopConfiguration()

    session = boto3.Session(
        aws_access_key_id=hc.get("key"),
        aws_secret_access_key=hc.get("secret_key")
    )
    s3_resource = session.resource('s3')

    bucket = 'bucket_name'
    key = path + model_name

    loaded_model = pickle.loads(s3_resource.Bucket(bucket).Object(key).get()['Body'].read())

    return loaded_model
