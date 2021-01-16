class StandardMetricsSQL:
    def __init__(self):

        self.EXPOSURE = dict(
        name = 'exposure',
        sql = """
        SELECT
          dw_users.user_id
          , DATE(COALESCE(guest_dw_users.joined_at,dw_users.joined_at)) as joined_date
          , DATE_PART(year, COALESCE(guest_dw_users.joined_at,dw_users.joined_at))::INT as cohort_year
          , dw_users.{segment_type} as segment
          , CASE
                WHEN dw_users.{segment_type} IN ({test_segments}) THEN 'TEST'
                WHEN dw_users.{segment_type} IN ({control_segments}) THEN 'CONTROL'
            ELSE 'NO TEST' END as test_group
          , CASE
                WHEN dw_users.gender IN ('male', 'male (hidden)') THEN 'male'
                WHEN dw_users.gender in ('female','female (hidden)') THEN 'female'
                ELSE 'unspecified'
            END AS gender
          , DATE(dw_users.buyer_activated_at) as buyer_activated_date
          , DATE(dw_users.lister_activated_at) as lister_activated_date
          , dw_users.gender as raw_gender
          , dw_users.reg_app
          , dw_users.reg_method
          , d_dates.full_dt AS event_date
          , MIN(dw_daily_user_events.event_date) AS first_date
        FROM analytics.dw_user_events_daily  AS dw_daily_user_events
        JOIN analytics.dw_users  AS dw_users ON dw_daily_user_events.user_id  = dw_users.user_id
        LEFT JOIN analytics.dw_users  AS guest_dw_users ON dw_users.guest_user_id = guest_dw_users.user_id and guest_dw_users.guest_user is true
        JOIN analytics.d_dates AS d_dates ON full_dt BETWEEN '{metric_start_date}' AND '{metric_end_date}'
          AND COALESCE(dw_users.delete_reason,'') not in ('guest_secured','desc_blacklist','website_blacklist') and dw_users.is_valid_user is true
          AND dw_daily_user_events.is_active
          AND dw_daily_user_events.is_valid_user
          --AND ((app_foreground>0 AND app<>'web') OR (app='web'))
          AND app in ({exposure_activity_apps})
          AND event_date BETWEEN '{exposure_start_date}' AND '{exposure_end_date}'
          AND dw_users.{segment_type} IN ({segments})
          AND dw_users.home_domain IN ({home_domain})
          {exposure_versions}

        GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12
        """,
        )

        self.ACTIVATIONS = dict(
        name = 'activations',
        sql = """
        SELECT
            dw_users.user_id
            , MIN(DATE(COALESCE(guest_dw_users.guest_joined_at,dw_users.joined_at))) as joined_date
            , MIN(DATEDIFF(day, COALESCE(guest_dw_users.guest_joined_at,dw_users.joined_at), dw_users.buyer_activated_at)) as days_to_buyer_activation
            , MIN(DATEDIFF(day, COALESCE(guest_dw_users.guest_joined_at,dw_users.joined_at), dw_users.lister_activated_at)) as days_to_lister_activation
            , MIN(DATEDIFF(day, COALESCE(guest_dw_users.guest_joined_at,dw_users.joined_at), activation.community_like_activated_at)) as days_to_liker_activation
            , MIN(DATEDIFF(day, COALESCE(guest_dw_users.guest_joined_at,dw_users.joined_at), dw_daily_user_events.event_date)) AS first_active_day_since_joined
        FROM analytics.dw_users  AS dw_users
        LEFT JOIN analytics.dw_guest_users_lookup  AS guest_dw_users ON dw_users.user_id = guest_dw_users.registered_user_id and coalesce(guest_dw_users.guest_reg_app,'') not in ('iphone','ipad')
        LEFT JOIN analytics.dw_users_cs AS activation ON activation.user_id = dw_users.user_id
        LEFT JOIN analytics.dw_user_events_daily  AS dw_daily_user_events
            ON dw_daily_user_events.user_id  = dw_users.user_id
            AND dw_daily_user_events.event_date BETWEEN '{metric_start_date}' AND '{metric_end_date}'
            AND DATEDIFF(day, COALESCE(guest_dw_users.guest_joined_at,dw_users.joined_at), dw_daily_user_events.event_date)>=1
            AND dw_daily_user_events.is_active
            AND dw_daily_user_events.is_valid_user
            --AND dw_daily_user_events.app IN ({metric_activity_apps})
        WHERE COALESCE(dw_users.delete_reason,'') not in ('guest_secured','desc_blacklist','website_blacklist') and dw_users.is_valid_user is true
        AND DATE(COALESCE(guest_dw_users.guest_joined_at,dw_users.joined_at)) BETWEEN '{metric_start_date}' AND '{metric_end_date}'
        AND dw_users.reg_app IN ({reg_apps})
        GROUP BY 1""",
        join = "activations.user_id = exposure.user_id AND activations.joined_date = exposure.event_date",
        metric_list = [
            """NULLIF(COUNT(distinct activations.user_id), 0) as "New Users" """,
            """CASE WHEN exposure.event_date <= DATE('{metric_end_date}') - 1 THEN 1.0*COUNT(DISTINCT CASE WHEN activations.first_active_day_since_joined=1 THEN activations.user_id END) END as "D2 Retained Users" """,
            """CASE WHEN exposure.event_date <= DATE('{metric_end_date}') - 1 THEN 1.0*COUNT(DISTINCT CASE WHEN activations.first_active_day_since_joined=1 THEN activations.user_id END)/NULLIF(COUNT(distinct activations.user_id),0) END as "D2 Retention" """,
            """CASE WHEN exposure.event_date <= DATE('{metric_end_date}') - 7 THEN 1.0*COUNT(DISTINCT CASE WHEN activations.first_active_day_since_joined BETWEEN 1 AND 6 THEN activations.user_id END)/NULLIF(COUNT(distinct activations.user_id),0) END as "d2-D7 Retention" """,
            """1.0*COUNT(DISTINCT CASE WHEN activations.days_to_buyer_activation=0 THEN activations.user_id END)/NULLIF(COUNT(distinct activations.user_id),0) as "D1 Buyer Activation" """,
            """CASE WHEN exposure.event_date <= DATE('{metric_end_date}') - 7 THEN 1.0*COUNT(DISTINCT CASE WHEN activations.days_to_buyer_activation<=6 THEN activations.user_id END)/NULLIF(COUNT(distinct activations.user_id),0)  END as "D7 Buyer Activation" """,
            """1.0*COUNT(DISTINCT CASE WHEN activations.days_to_lister_activation=0 THEN activations.user_id END)/NULLIF(COUNT(distinct activations.user_id),0) as "D1 Lister Activation" """,
            """CASE WHEN exposure.event_date <= DATE('{metric_end_date}') - 7 THEN 1.0*COUNT(DISTINCT CASE WHEN activations.days_to_lister_activation<=6  THEN activations.user_id END)/NULLIF(COUNT(distinct activations.user_id),0)  END as "D7 Lister Activation" """,
            """COUNT(DISTINCT CASE WHEN activations.days_to_liker_activation=0 THEN activations.user_id END) as "D1 Likers" """,
            """1.0*COUNT(DISTINCT CASE WHEN activations.days_to_liker_activation=0 THEN activations.user_id END)/NULLIF(COUNT(distinct activations.user_id),0) as "D1 Liker Activation" """,
            """CASE WHEN exposure.event_date <= DATE('{metric_end_date}') - 7 THEN 1.0*COUNT(DISTINCT CASE WHEN activations.days_to_liker_activation<=6  THEN activations.user_id END)/NULLIF(COUNT(distinct activations.user_id),0)  END as "D7 Liker Activation" """,
            ]
        )

        self.BUYER_NEW_USERS = dict(
        name = 'buyer_new_users',
        sql = """
        SELECT
            dw_users.user_id
            , DATE(COALESCE(guest_dw_users.joined_at,dw_users.joined_at)) as joined_date
            , COUNT(DISTINCT CASE WHEN DATEDIFF(day, DATE(COALESCE(guest_dw_users.joined_at,dw_users.joined_at)), dw_orders.booked_at) = 0 THEN dw_orders.buyer_id END) as D1_Buyers
            , COUNT(DISTINCT CASE WHEN DATEDIFF(day, DATE(COALESCE(guest_dw_users.joined_at,dw_users.joined_at)), dw_orders.booked_at) BETWEEN 0 AND 6 THEN dw_orders.buyer_id END) as D7_Buyers
            , COUNT(DISTINCT CASE WHEN DATEDIFF(day, DATE(COALESCE(guest_dw_users.joined_at,dw_users.joined_at)), dw_orders.booked_at) = 0 THEN dw_orders.order_id END) as D1_Orders
            , COUNT(DISTINCT CASE WHEN DATEDIFF(day, DATE(COALESCE(guest_dw_users.joined_at,dw_users.joined_at)), dw_orders.booked_at) BETWEEN 0 AND 6 THEN dw_orders.order_id END) as D7_Orders
            , SUM(CASE WHEN DATEDIFF(day, DATE(COALESCE(guest_dw_users.joined_at,dw_users.joined_at)), dw_orders.booked_at) = 0 THEN 0.01*dw_orders.order_gmv END) as D1_GMV """ + """
            , SUM(CASE WHEN DATEDIFF(day, DATE(COALESCE(guest_dw_users.joined_at,dw_users.joined_at)), dw_orders.booked_at) BETWEEN 0 AND 6 THEN 0.01*dw_orders.order_gmv END) as D7_GMV """ + """
        FROM analytics.dw_users  AS dw_users
        LEFT JOIN analytics.dw_users  AS guest_dw_users ON dw_users.guest_user_id = guest_dw_users.user_id and guest_dw_users.guest_user is true and guest_dw_users.reg_app = 'web'
        LEFT JOIN analytics.dw_orders AS dw_orders ON dw_users.user_id = dw_orders.buyer_id
            AND DATE(dw_orders.booked_at) BETWEEN '{metric_start_date}' AND '{metric_end_date}'
            AND dw_orders.is_valid_order = TRUE
        WHERE COALESCE(dw_users.delete_reason,'') not in ('guest_secured','desc_blacklist','website_blacklist') and dw_users.is_valid_user is true
        AND DATE(COALESCE(guest_dw_users.joined_at,dw_users.joined_at)) BETWEEN '{metric_start_date}' AND '{metric_end_date}'
        AND dw_users.reg_app IN ({reg_apps})
        GROUP BY 1,2""",
        join = "buyer_new_users.user_id = exposure.user_id AND buyer_new_users.joined_date = exposure.event_date",
        metric_list = [
            """SUM(D1_Buyers) as "D1 Buyers" """,
            """NULLIF(SUM(CASE WHEN exposure.event_date <= DATE('{metric_end_date}') - 7 THEN D7_Buyers END),0) as "D7 Buyers" """,
            """SUM(D1_Orders) as "D1 Orders" """,
            """NULLIF(SUM(CASE WHEN exposure.event_date <= DATE('{metric_end_date}') - 7 THEN D7_Orders END),0) as "D7 Orders" """,
            """SUM(D1_GMV) as "D1 GMV" """,
            """NULLIF(SUM(CASE WHEN exposure.event_date <= DATE('{metric_end_date}') - 7 THEN D7_GMV END),0) as "D7 GMV" """,
            """1.0*SUM(D1_Orders)/NULLIF(COUNT(distinct buyer_new_users.user_id),0) as "D1 Orders per New Users" """,
            """NULLIF(1.0*SUM(CASE WHEN exposure.event_date <= DATE('{metric_end_date}') - 7 THEN D7_Orders END),0)/NULLIF(COUNT(distinct buyer_new_users.user_id),0) as "D7 Orders per New Users" """,
            """1.0*SUM(D1_GMV)/NULLIF(COUNT(distinct buyer_new_users.user_id),0) as "D1 GMV per New Users" """,
            """NULLIF(1.0*SUM(CASE WHEN exposure.event_date <= DATE('{metric_end_date}') - 7 THEN D7_GMV END),0)/NULLIF(COUNT(distinct buyer_new_users.user_id),0) as "D7 GMV per New Users" """,
            ]
        )


        self.BUYER_ORDERS = dict(
        name = 'buyer_orders',
        sql = """
        SELECT
          dw_orders.buyer_id as user_id
          , DATE(dw_orders.booked_at) as event_date
          , sum(0.01*order_gmv) as gmv
          , count(0) as orders
          , COUNT(CASE WHEN dw_orders.cancelled_on IS NOT NULL THEN 1 END) as canceled_orders
          , sum(CASE WHEN order_number > 1 THEN 0.01*order_gmv END) as repeat_gmv
          , count(CASE WHEN order_number > 1 THEN 1 END) as repeat_orders
          , COUNT(CASE WHEN order_number = 1 THEN 1 END) AS count_first_orders
          , SUM(CASE WHEN order_number = 1 THEN 0.01*order_gmv END) AS first_order_gmv
          , sum(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='buyer' THEN 0.01*order_gmv END) as gmv_buyer_offer
          , count(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='buyer' THEN 1 END) as orders_buyer_offer
          , sum(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='buyer' AND order_number > 1 THEN 0.01*order_gmv END) as repeat_gmv_buyer_offer
          , count(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='buyer' AND order_number > 1 THEN 1 END) as repeat_orders_buyer_offer
          , sum(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='seller' THEN 0.01*order_gmv END) as gmv_seller_offer
          , sum(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='seller' AND order_number > 1 THEN 0.01*order_gmv END) as repeat_gmv_seller_offer
          , count(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='seller' THEN 1 END) as orders_seller_offer
          , count(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='seller' AND order_number > 1 THEN 1 END) as repeat_orders_seller_offer
          , count(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='seller' AND order_number = 1 THEN 1 END) as first_orders_seller_offer
          , sum(CASE WHEN dw_orders.offer_id is NULL THEN 0.01*order_gmv END) as gmv_buy_now
          , count(CASE WHEN dw_orders.offer_id is NULL THEN 1 END) as orders_buy_now
          , count(CASE WHEN dw_orders.offer_id is NULL AND buyer.guest_user = TRUE or (buyer.guest_user_id is not null
                  and (DATE(buyer.buyer_activated_at ))= (DATE(dw_orders.booked_at ))
                  and buyer.buyer_activated_at<buyer.joined_at) THEN 1 END) as orders_buy_now_guest
          , count(CASE WHEN dw_orders.offer_id is NULL AND NOT(buyer.guest_user = TRUE or (buyer.guest_user_id is not null
                  and (DATE(buyer.buyer_activated_at ))= (DATE(dw_orders.booked_at ))
                  and buyer.buyer_activated_at<buyer.joined_at)) THEN 1 END) as orders_buy_now_not_guest
          , count(CASE WHEN dw_orders.offer_id is NULL AND order_number = 1 THEN 1 END) as first_order_buy_now
          , count(CASE WHEN dw_orders.offer_id is NULL AND order_number > 1 THEN 1 END) as repeat_order_buy_now
          , count(DISTINCT CASE WHEN dw_orders.offer_id is NULL AND order_number > 1 THEN dw_orders.buyer_id END) as repeat_buy_now_buyers
          , sum(CASE WHEN dw_orders.offer_id is NULL AND order_number > 1 THEN 0.01*order_gmv END) as repeat_gmv_buy_now
          , count(CASE WHEN dw_orders.offer_id is NULL AND order_number = 1 AND (dw_orders.payment_method NOT ILIKE 'Android Pay') THEN 1 END) as first_order_buy_now_not_android_pay
          , count(CASE WHEN dw_orders.offer_id is NULL AND (dw_orders.payment_method ILIKE '%android%' OR dw_orders.payment_method ILIKE '%google%' ) THEN 1 END) as orders_buy_now_google
          , count(CASE WHEN dw_orders.offer_id is NULL AND (dw_orders.payment_method ILIKE '%apple%') THEN 1 END) as orders_buy_now_apple
          , count(CASE WHEN dw_orders.offer_id is NULL AND (dw_orders.payment_method ILIKE '%credit card%') THEN 1 END) as orders_buy_now_credit_card
          , count(CASE WHEN dw_orders.offer_id is NULL AND (dw_orders.payment_method ILIKE '%venmo%') THEN 1 END) as orders_buy_now_venmo
          , count(CASE WHEN dw_orders.offer_id is NULL AND (dw_orders.payment_method ILIKE '%paypal%') THEN 1 END) as orders_buy_now_paypal
          , sum(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='seller' AND dw_offers.offer_broadcast_id is NOT NULL THEN 0.01*order_gmv END) as seller_otl_gmv
          , sum(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='seller' AND dw_offers.offer_broadcast_id is NULL THEN 0.01*order_gmv END) as seller_direct_offer_gmv
          , count(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='seller' AND dw_offers.offer_broadcast_id is NOT NULL THEN 1 END) as seller_otl_orders
          , count(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='seller' AND dw_offers.offer_broadcast_id is NULL THEN 1 END) as seller_direct_offer_orders
          , COUNT(DISTINCT CASE WHEN (bundle_orders.bundle_id IS NOT NULL or dw_orders.is_bundle = TRUE) THEN dw_orders.order_id ELSE NULL END ) as bundle_orders
        FROM analytics.dw_orders as dw_orders
        LEFT JOIN analytics.dw_users  AS buyer ON buyer.user_id = dw_orders.buyer_id
        LEFT JOIN analytics.dw_guest_users_lookup  AS guest_buyer ON buyer.user_id = guest_buyer.registered_user_id and coalesce(guest_buyer.guest_reg_app,'') not in ('iphone','ipad')
        LEFT JOIN analytics.dw_fx_table  AS order_foreign_exchange ON (DATE(dw_orders.booked_at )) = (DATE(order_foreign_exchange.valid_from )) and order_foreign_exchange.target_currency='USD'
        LEFT JOIN analytics.dw_offers as dw_offers ON dw_orders.offer_id = dw_offers.offer_id
        LEFT JOIN analytics.dw_bundles_v3  AS bundle_orders ON bundle_orders.order_id = dw_orders.order_id
            and bundle_orders.buyer_id = dw_orders.buyer_id
            and bundle_orders.seller_id = dw_orders.seller_id
        WHERE DATE(dw_orders.booked_at) BETWEEN '{metric_start_date}' AND '{metric_end_date}'
          AND dw_orders.booked_at >'01-01-2015'
          AND dw_orders.is_valid_order = TRUE
        AND app IN ({metric_activity_apps})
        GROUP BY 1,2
        """ ,
        join = "buyer_orders.user_id = exposure.user_id AND buyer_orders.event_date = exposure.event_date",
        metric_list = [
            """COUNT(DISTINCT buyer_orders.user_id) as Buyers""",
            """SUM(repeat_buy_now_buyers) as "Repeat Buy Now Buyers" """,
            """SUM(buyer_orders.orders) as Orders""",
            """SUM(buyer_orders.canceled_orders) as "Orders - Canceled" """,
            """SUM(buyer_orders.repeat_orders) as "Repeat Orders" """,
            """SUM(buyer_orders.count_first_orders) as "First Time Buyers" """,
            """SUM(buyer_orders.first_order_gmv) as "First Order GMV" """,
            """SUM(buyer_orders.gmv) as GMV""",
            """SUM(buyer_orders.repeat_gmv) as "Repeat GMV" """,
            """COUNT(DISTINCT CASE WHEN DATEDIFF(day, exposure.joined_date, buyer_orders.event_date) BETWEEN 0 AND 6 THEN buyer_orders.user_id END) as "Weekly New User Buyer" """,
            """SUM(CASE WHEN DATEDIFF(day, exposure.joined_date, buyer_orders.event_date) BETWEEN 0 AND 6 THEN buyer_orders.orders END) as "Weekly New User Orders" """,
            """SUM(CASE WHEN DATEDIFF(day, exposure.joined_date, buyer_orders.event_date) BETWEEN 0 AND 6 THEN buyer_orders.gmv END) as "Weekly New User GMV" """,
            """COUNT(DISTINCT CASE WHEN DATEDIFF(day, exposure.joined_date, buyer_orders.event_date) >=7 THEN buyer_orders.user_id END) as "Weekly Current User Buyer" """,
            """SUM(CASE WHEN DATEDIFF(day, exposure.joined_date, buyer_orders.event_date)  >=7 THEN buyer_orders.orders END) as "Weekly Current User Orders" """,
            """SUM(CASE WHEN DATEDIFF(day, exposure.joined_date, buyer_orders.event_date)  >=7 THEN buyer_orders.gmv END) as "Weekly Current User GMV" """,
            """SUM(buyer_orders.orders_buyer_offer) as "Orders - Buyer Offer" """,
            """SUM(buyer_orders.gmv_buyer_offer) as "GMV - Buyer Offer" """,
            """SUM(buyer_orders.repeat_orders_buyer_offer) as "Repeat Orders - Buyer Offer" """,
            """SUM(buyer_orders.repeat_gmv_buyer_offer) as "Repeat GMV - Buyer Offer" """,
            """SUM(buyer_orders.orders_seller_offer) as "Orders - Seller Offer" """,
            """SUM(buyer_orders.repeat_orders_seller_offer) as "Repeat Orders - Seller Offer" """,
            """SUM(buyer_orders.first_orders_seller_offer) as "First Orders - Seller Offer" """,
            """SUM(buyer_orders.gmv_seller_offer) as "GMV - Seller Offer" """,
            """SUM(buyer_orders.repeat_gmv_seller_offer) as "Repeat GMV - Seller Offer" """,
            """SUM(buyer_orders.orders_buy_now) as "Orders - Buy Now" """,
            """SUM(buyer_orders.orders_buy_now_guest) as "Orders - Guest Buy Now" """,
            """SUM(buyer_orders.orders_buy_now_not_guest) as "Orders - Non-Guest Buy Now" """,
            """SUM(buyer_orders.first_order_buy_now) as "First Order - Buy Now" """,
            """SUM(buyer_orders.repeat_order_buy_now) as "Repeat Order - Buy Now" """,
            """SUM(buyer_orders.first_order_buy_now_not_android_pay) as "First Order - Buy Now Not Android Pay" """,
            """SUM(buyer_orders.gmv_buy_now) as "GMV - Buy Now" """,
            """SUM(buyer_orders.repeat_gmv_buy_now) as "Repeat GMV - Buy Now" """,
            """SUM(buyer_orders.orders_buy_now_google) as "Buy Now Orders - Google" """,
            """SUM(buyer_orders.orders_buy_now_apple) as "Buy Now Orders - Apple" """,
            """SUM(buyer_orders.orders_buy_now_credit_card) as "Buy Now Orders - Credit Card" """,
            """SUM(buyer_orders.orders_buy_now_venmo) as "Buy Now Orders - Venmo" """,
            """SUM(buyer_orders.orders_buy_now_paypal) as "Buy Now Orders - Paypal" """,
            """SUM(buyer_orders.seller_otl_gmv) as "Seller OTL GMV" """,
            """SUM(buyer_orders.seller_direct_offer_gmv) as "Seller Direct Offer GMV" """,
            """SUM(buyer_orders.seller_otl_orders) as "Seller OTL Orders" """,
            """SUM(buyer_orders.seller_direct_offer_orders) as "Seller Direct Offer Orders" """,
            """SUM(buyer_orders.bundle_orders) as "Bundle Orders" """,
            ]
        )
        # CA metrics changes - Seller GMV buy now, seller gmv seller offer, seller_gmv_buyer_offer, seller_gmv
        self.SELLER_ORDERS = dict(
        name = 'seller_orders',
        sql = """
        SELECT
          dw_orders.seller_id as user_id
          , DATE(dw_orders.booked_at) as event_date
          , sum(CASE
                WHEN (upper(dw_orders.origin_domain)) ILIKE 'CA' THEN dw_orders.order_gmv*0.01*foreign_exchange.exchange_rate
                WHEN (upper(dw_orders.origin_domain)) NOT ILIKE 'CA' THEN dw_orders.order_gmv*0.01
              END) as seller_gmv
          , count(0) as seller_orders
          , COUNT(CASE WHEN order_number = 1 THEN 1 END) AS seller_count_first_orders
          , sum(CASE
                WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='buyer' AND (upper(dw_orders.origin_domain)) ILIKE 'CA' THEN dw_orders.order_gmv*0.01*foreign_exchange.exchange_rate
                WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='buyer' AND (upper(dw_orders.origin_domain)) NOT ILIKE 'CA' THEN dw_orders.order_gmv*0.01
              END) as seller_gmv_buyer_offer
          , count(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='buyer' THEN 1 END) as seller_orders_buyer_offer
          , sum(CASE
                WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='seller' AND (upper(dw_orders.origin_domain)) ILIKE 'CA' THEN dw_orders.order_gmv*0.01*foreign_exchange.exchange_rate
                WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='seller' AND (upper(dw_orders.origin_domain)) NOT ILIKE 'CA' THEN dw_orders.order_gmv*0.01
              END) as seller_gmv_seller_offer
          , count(CASE WHEN dw_orders.offer_id is NOT NULL AND offer_initiated_by='seller' THEN 1 END) as seller_orders_seller_offer
          , COALESCE(SUM(
              CASE
                WHEN dw_orders.offer_id is NULL AND (upper(dw_orders.origin_domain)) ILIKE 'CA' THEN dw_orders.order_gmv*0.01*foreign_exchange.exchange_rate
                WHEN dw_orders.offer_id is NULL AND (upper(dw_orders.origin_domain)) NOT ILIKE 'CA' THEN dw_orders.order_gmv*0.01
              END), 0) AS seller_gmv_buy_now
          , count(CASE WHEN dw_orders.offer_id is NULL THEN 1 END) as seller_orders_buy_now
        FROM analytics.dw_orders as dw_orders
        LEFT JOIN analytics.dw_offers as dw_offers ON dw_orders.offer_id = dw_offers.offer_id
        LEFT JOIN analytics.dw_fx_table  AS foreign_exchange ON (DATE(dw_orders.booked_at )) = (DATE(foreign_exchange.valid_from )) AND foreign_exchange.target_currency='USD'
        WHERE DATE(dw_orders.booked_at) BETWEEN '{metric_start_date}' AND '{metric_end_date}'
          AND dw_orders.booked_at >'01-01-2015'
          AND dw_orders.is_valid_order = TRUE
        AND app IN ({metric_activity_apps})
        GROUP BY 1,2
        """ ,
        join = "seller_orders.user_id = exposure.user_id AND seller_orders.event_date = exposure.event_date",
        metric_list = [
            """COUNT(DISTINCT seller_orders.user_id) as Sellers""",
            """SUM(seller_orders.seller_orders) as "Seller Orders" """,
            #"""SUM(seller_orders.seller_count_first_orders) as "Seller First Orders" """,
            """SUM(seller_orders.seller_gmv) as "Seller GMV" """,
            """SUM(seller_orders.seller_orders_buyer_offer) as "Seller Orders - Buyer Offer" """,
            """SUM(seller_orders.seller_gmv_buyer_offer) as "Seller GMV - Buyer Offer" """,
            """SUM(seller_orders.seller_orders_seller_offer) as "Seller Orders - Seller Offer" """,
            """SUM(seller_orders.seller_gmv_seller_offer) as "Seller GMV - Seller Offer" """,
            """SUM(seller_orders.seller_orders_buy_now) as "Seller Orders - Buy Now" """,
            """SUM(seller_orders.seller_gmv_buy_now) as "Seller GMV - Buy Now" """,
            ]
        )

        self.BUYER_OFFERS = dict(
        name = 'buyer_offers',
        sql = """
        SELECT
            offers.buyer_id as user_id
            ,DATE(offers.created_at) as event_date
            ,COUNT(CASE WHEN offers.offer_initiated_by='buyer' THEN 1 END) AS buyer_initiated_offer_sent
            ,COUNT(CASE WHEN offers.offer_initiated_by='buyer' AND orders.order_number > 1 THEN 1 END) AS repeat_buyer_initiated_offer_sent
            ,COUNT(CASE WHEN offers.offer_initiated_by='seller' THEN 1 END) AS seller_initiated_offer_received
            ,COUNT(DISTINCT CASE WHEN offers.offer_broadcast_id is not null THEN offers.offer_id END) AS otl_offers_received
            ,COUNT(CASE WHEN (datediff('hours', offers.created_at, orders.booked_at) <= 48) AND offers.offer_broadcast_id is not null THEN 1 ELSE NULL END) AS buyer_orders_48_hours_of_otl
            ,COUNT(CASE WHEN (datediff('hours', offers.created_at, orders.booked_at) <= 24) AND offers.offer_broadcast_id is not null THEN 1 ELSE NULL END) AS buyer_orders_24_hours_of_otl
            ,COALESCE(SUM((
              CASE
                WHEN offers.offer_broadcast_id is not null AND (upper(orders.origin_domain)) ILIKE 'CA' then orders.order_gmv*0.01*foreign_exchange.exchange_rate
                WHEN offers.offer_broadcast_id is not null AND (upper(orders.origin_domain)) NOT ILIKE 'CA' THEN orders.order_gmv*0.01
                END)
            ), 0) AS buyer_otl_order_gmv
        FROM analytics.dw_offers as offers
        LEFT JOIN analytics.dw_orders  AS orders ON offers.offer_id = orders.offer_id and offers.order_id = orders.order_id
        LEFT JOIN analytics.dw_fx_table  AS foreign_exchange ON (DATE(orders.booked_at )) = (DATE(foreign_exchange.valid_from )) AND foreign_exchange.target_currency='USD'
        WHERE DATE(offers.created_at) BETWEEN '{metric_start_date}' AND '{metric_end_date}'
        GROUP BY 1,2
        """ ,
        join = "buyer_offers.user_id = exposure.user_id AND buyer_offers.event_date = exposure.event_date",
        metric_list = [
          """SUM(buyer_initiated_offer_sent) as "Buyer Initiated Offer Sent" """,
          """SUM(repeat_buyer_initiated_offer_sent) as "Repeat Buyer Initiated Offer Sent" """,
          """SUM(seller_initiated_offer_received) as "Seller Initiated Offer Received" """,
          """SUM(otl_offers_received) as "OTL Offers Received" """,
          """SUM(buyer_orders_48_hours_of_otl) as "Buyer OTL 48hrs Orders" """,
          """SUM(buyer_orders_24_hours_of_otl) as "Buyer OTL 24hrs Orders" """,
          """SUM(buyer_otl_order_gmv) as "Buyer OTL Order GMV" """,
          ]
        )

        self.SELLER_OFFERS = dict(
        name = 'seller_offers',
        sql = """
        SELECT
            offers.lister_id as user_id
            ,DATE(offers.created_at) as event_date
            ,COUNT(CASE WHEN offers.offer_initiated_by='seller' THEN 1 END) AS seller_initiated_offer_sent
            ,COUNT(DISTINCT CASE WHEN offers.offer_broadcast_id is not null THEN offers.offer_broadcast_id END) AS otl_broadcasts
            ,COUNT(DISTINCT CASE WHEN offers.offer_broadcast_id is not null THEN offers.offer_id END) AS otl_offers_sent
            ,COUNT(CASE WHEN (datediff('hours', offers.created_at, orders.booked_at) <= 48) AND offers.offer_broadcast_id is not null THEN 1 ELSE NULL END) AS orders_within_48_hours_of_otl
            ,COALESCE(SUM((CASE WHEN (upper(orders.origin_domain)) ILIKE 'CA' then orders.order_gmv*0.01*foreign_exchange.exchange_rate else orders.order_gmv*0.01 end) ), 0) AS offer_order_gmv
            ,COALESCE(SUM((
              CASE
                WHEN offers.offer_broadcast_id is not null AND (upper(orders.origin_domain)) ILIKE 'CA' then orders.order_gmv*0.01*foreign_exchange.exchange_rate
                WHEN offers.offer_broadcast_id is not null AND (upper(orders.origin_domain)) NOT ILIKE 'CA' THEN orders.order_gmv*0.01
                END)
              ), 0) AS otl_order_gmv
            ,COUNT(DISTINCT CASE WHEN offers.offer_initiated_by='seller' AND offers.offer_broadcast_id is NULL THEN offers.offer_id END) as seller_direct_offer_sent
        FROM analytics.dw_offers as offers
        LEFT JOIN analytics.dw_orders  AS orders ON offers.offer_id = orders.offer_id and offers.order_id = orders.order_id
        LEFT JOIN analytics.dw_fx_table  AS foreign_exchange ON (DATE(orders.booked_at )) = (DATE(foreign_exchange.valid_from )) AND foreign_exchange.target_currency='USD'
        WHERE DATE(offers.created_at) BETWEEN '{metric_start_date}' AND '{metric_end_date}'
        GROUP BY 1,2
        """ ,
        join = "seller_offers.user_id = exposure.user_id AND seller_offers.event_date = exposure.event_date",
        metric_list = [
            """SUM(seller_initiated_offer_sent) as "Seller Initiated Offer Sent" """,
            """SUM(otl_broadcasts) as "OTL Broadcasts" """,
            """SUM(otl_offers_sent) as "OTL Offers Sent" """,
            """SUM(orders_within_48_hours_of_otl) as "OTL 48hrs Orders" """,
            """SUM(offer_order_gmv) as "Offer Order GMV" """,
            """SUM(otl_order_gmv) as "OTL Order GMV" """,
            """SUM(seller_direct_offer_sent) as "Seller Direct Offer Sent" """,
            ]
        )

        self.LISTINGS = dict(
        name = "listings",
        sql = """
        SELECT
            seller_id as user_id
            , DATE(created_at) AS event_date
            , COUNT(1) as listings_created
        FROM analytics.dw_listings
        WHERE DATE(created_at) BETWEEN '{metric_start_date}' AND '{metric_end_date}'
            AND (((dw_listings.listing_price*.01) <= 75000 AND ( dw_listings.create_source_type IS NULL  OR dw_listings.parent_listing_id IS NOT NULL)) = TRUE
                   AND (dw_listings.listing_type = 'child_item' OR dw_listings.create_source_type IS NULL))
            AND dw_listings.parent_listing_id IS NULL
        AND app IN ({metric_activity_apps})
        GROUP BY 1,2
        """,
        join = "listings.user_id = exposure.user_id AND listings.event_date = exposure.event_date",
        metric_list = [
            """SUM(listings_created) as Listings""",
            """COUNT(DISTINCT listings.user_id) as Listers""",
            ]
        )

        self.EVENTS = dict(
            name = 'events',
        sql = """
        SELECT
          dw_daily_user_events.user_id
          , event_date
          , MAX(CASE WHEN dw_daily_user_events.is_active AND dw_daily_user_events.is_valid_user THEN 1 ELSE 0 END) as is_active_user
          , SUM(community_likes) as community_likes
          , SUM(self_likes) as self_likes
          , SUM(coalesce(party_community_shares,0) + coalesce(followers_community_shares,0)) as community_shares
          , SUM(coalesce(party_self_shares,0) + coalesce(followers_self_shares,0)) as self_shares
          , SUM(coalesce(community_listings_viewed,0)) as community_listings_viewed
          , SUM(coalesce(self_listings_viewed,0)) as self_listings_viewed
          , SUM(users_manually_followed) as manual_follows
          , SUM(users_auto_followed) as auto_follows
          , SUM(community_comments) as community_comments
          , SUM(closet_viewed) as closet_viewed
          , SUM(keyword_searches) as keyword_searches
          , SUM(page_views) as page_views
        FROM analytics.dw_user_events_daily  AS dw_daily_user_events
        WHERE event_date BETWEEN '{metric_start_date}' AND '{metric_end_date}'
          AND app IN ({metric_activity_apps})
        GROUP BY 1,2
        """,
        join = "events.user_id = exposure.user_id AND events.event_date = exposure.event_date",
        metric_list = [
            """COUNT(DISTINCT CASE WHEN is_active_user = 1 THEN events.event_date||events.user_id END) as DAU""",
            """SUM(community_likes) as Likes""",
            """SUM(community_shares) as "Community Shares" """,
            """SUM(self_shares) as "Self Shares" """,
            """SUM(community_listings_viewed) as "Listing Detail Views" """,
            """SUM(self_listings_viewed) as "Self Listing Detail Views" """,
            """SUM(manual_follows) as "Manual Follows" """,
            """SUM(auto_follows) as "Auto Follows" """,
            """SUM(community_comments) as Comment""",
            """SUM(closet_viewed) as "Closets Viewed" """,
            """SUM(keyword_searches) as "Keyword Searches" """,
            """SUM(page_views) as "Page Views" """,
            """COUNT(DISTINCT CASE WHEN community_listings_viewed>0 THEN events.user_id END) as "Listing Detail Viewers" """,
            """COUNT(DISTINCT CASE WHEN community_likes>0 THEN events.user_id END) as "Likers" """,
            """COUNT(DISTINCT CASE WHEN community_comments>0 THEN events.user_id END) as Commenters""",
            """COUNT(DISTINCT CASE WHEN community_shares>0 THEN events.user_id END) as "Community Sharers" """,
            """COUNT(DISTINCT CASE WHEN keyword_searches>0 THEN events.user_id END) as "Keyword Searchers" """,
            """COUNT(DISTINCT CASE WHEN manual_follows>0 THEN events.user_id END) as "Manual Followers" """,
          ]
        )

        self.BUNDLES = dict(
            name = 'bundles',
        sql = """
        SELECT
            buyer_id as user_id
            , event_date
            , COUNT(CASE WHEN activity_type = 'item_added' THEN 1 END) as add_to_bundles
            , COUNT(CASE WHEN activity_type = 'bundle_created' THEN 1 END) as bundle_created
        FROM analytics.dw_bundles_activity_cs
        WHERE event_date BETWEEN '{metric_start_date}' AND '{metric_end_date}'
        AND activity_type IN ('item_added', 'bundle_created')
        AND user_interaction_context = 'buyer'
        GROUP BY 1,2
        """,
        join = "bundles.user_id = exposure.user_id AND bundles.event_date = exposure.event_date",
        metric_list = [
            """SUM(add_to_bundles) as "Add to Bundles" """,
            """SUM(bundle_created) as "Bundle Created" """,
          ]
        )

        self.INCEPTION = dict(
            name = 'bundles',
        sql = """
        SELECT
            user_id
            , event_date
            , SUM(sessions) AS sessions
            , SUM(timespent) AS timespent
        FROM analytics_scratch.user_sessions_daily
        WHERE event_date BETWEEN '{metric_start_date}' AND '{metric_end_date}'
          AND app_type IN ({metric_activity_apps})
        GROUP BY 1,2
        """,
        join = "bundles.user_id = exposure.user_id AND bundles.event_date = exposure.event_date",
        metric_list = [
            """SUM(sessions) AS sessions""",
            """SUM(timespent) AS timespent""",
          ]
        )

        self.LOOKER_ORDERS = dict(
          name = 'looker_orders',
          look_id = "12153",
          sql="""""",
          join = "looker_orders.buyer_id = exposure.user_id AND looker_orders.booked_date = exposure.event_date",
		      metric_list = [
            """SUM(buyer_orders.orders) as Looker_Orders""",
            """SUM(buyer_orders.gmv) as Looker_GMV"""
            ],
          replace = 'AS dw_orders.'
        )

        self.LIGHTSPEED_METRICS = [
            #COMMERCE
            'gmv',
            'orders',
            'buyers',
            'listers',
            'listings',
            'gmv per orders',
            'dau',
            'gmv per dau',
            'buyers per dau',
            'listers per dau',
            'listings per dau',
            'listings per listers',
            'orders - buyer offers',
            'orders - buy now',
            'orders - seller offers',

            #social
            'likes',
            'community shares',
            'self shares',

            #growth
            'new users',
            'd2 retention',
            'd2-d7 retention',
            'd1 buyer activation',
            'd1 lister activation',
            'd1 liker activation',
            'd7 buyer activation',
            'd7 lister activation',
            'd7 liker activation',

            #inception
            'sessions',
            'timespent',
            ]
