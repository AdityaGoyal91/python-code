def graph_daily_error(data_df, x_name, actual_y_name, pred_y_name, group_list, agg_func_list, graph_name, y_name):

    grouped_df = (
        data_df
            .groupby(
                group_list,
                as_index = False
            )
            .agg(agg_func_list)
    )
    
    grouped_df['gr_error_sq'] = (grouped_df[pred_y_name] - grouped_df[actual_y_name]) * (grouped_df[pred_y_name] - grouped_df[actual_y_name])
    grouped_rmse = math.sqrt((grouped_df['gr_error_sq'].sum()/grouped_df['gr_error_sq'].size))

    sns.set(font_scale = 1.5)

    fig, ax = pyplot.subplots(figsize = (14,6))

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.plot(grouped_df[x_name],grouped_df[actual_y_name])
    plt.plot(grouped_df[x_name],grouped_df[pred_y_name])

    ax.set_title(graph_name + ' Actual ' + y_name + ' vs Predicted ' + y_name, fontsize = 18)
    ax.legend(labels = ['actual','prediction'])
    ax.set_ylabel(y_name, fontsize = 16)
    ax.set_xlabel('Date', fontsize = 16)
    z.showplot(plt)
    return grouped_rmse, grouped_df
