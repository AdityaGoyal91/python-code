def save_model_image(model_to_save, model_name, location=):

    hc = sc._jsc.hadoopConfiguration()

    session = boto3.Session(
        aws_access_key_id="key"
        aws_secret_access_key="secret_key"
    )

    s3_resource = session.resource('s3')

    model_image = pickle.dumps(model_to_save)

    bucket = 'bucket_name'
    key = path + model_name

    model_obj = s3_resource.Object(bucket, key)
    model_obj.put(Body=model_image)

    print("Model stored in S3, please go check now")
