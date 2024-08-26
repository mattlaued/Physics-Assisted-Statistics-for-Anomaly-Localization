from AL_methods import *

import time

def main():
    data_dir = "../Data/Data_Power/"
    training_path = "Event 3.CSV"
    df = pd.read_csv(data_dir + training_path, skiprows=[0,1,2,4])

    # remove first 0.25 seconds to allow system to stabilise before using data
    data = preprocess(df, time=0.25e6, remove_neutral=True)

    # can check data corr
    corr_training = data.corr()
    corr_training.head()

    window = 320
    w = 1
    cols = data.columns

    corr_fn = correlation_pair
    thresholds_corr, corr_multi_list = get_threshold_stat_only(data, corr_fn, cols,
                  window=window, w=w, adj_list=adj_list, quantile=1, axis=1, stat_list=None, plot=True, collate_fn="sum")
    thresholds_corr_99, corr_multi_list = get_threshold_stat_only(data, corr_fn, cols,
                  window=window, w=w, adj_list=adj_list, quantile=0.99, axis=1, stat_list=corr_multi_list, collate_fn="sum")
    thresholds_corr_95, corr_multi_list = get_threshold_stat_only(data, corr_fn, cols,
                  window=window, w=w, adj_list=adj_list, quantile=0.95, axis=1, stat_list=corr_multi_list, collate_fn="sum")

    var_fn = var_pair
    thresholds_var, var_multi_list = get_threshold_stat_only(data, var_fn, cols,
                  window=window, w=w, adj_list=adj_list, quantile=1, axis=1, stat_list=None, plot=True, collate_fn="sum")
    thresholds_var_99, var_multi_list = get_threshold_stat_only(data, var_fn, cols,
                  window=window, w=w, adj_list=adj_list, quantile=0.99, axis=1, stat_list=var_multi_list, collate_fn="sum")
    thresholds_var_95, var_multi_list = get_threshold_stat_only(data, var_fn, cols,
                  window=window, w=w, adj_list=adj_list, quantile=0.95, axis=1, stat_list=var_multi_list, collate_fn="sum")


    # with open("models/" + f"thresholds_corr_{topology}.npy", 'rb') as f:
    #     thresholds_corr = np.load(f)
    # with open("models/" + f"thresholds_corr_99_{topology}.npy", 'rb') as f:
    #     thresholds_corr_99 = np.load(f)
    # with open("models/" + f"thresholds_corr_95_{topology}.npy", 'rb') as f:
    #     thresholds_corr_95 = np.load(f)
    #
    # with open("models/" + f"thresholds_var_{topology}.npy", 'rb') as f:
    #     thresholds_var = np.load(f)
    # with open("models/" + f"thresholds_var_99_{topology}.npy", 'rb') as f:
    #     thresholds_var_99 = np.load(f)
    # with open("models/" + f"thresholds_var_95_{topology}.npy", 'rb') as f:
    #     thresholds_var_95 = np.load(f)

    # Testing

    data_dir = "../Data/Data_Power/Sep2023Data/"
    test_path = "Event4_NORMALIZED.xlsx"
    df_test = pd.read_excel(data_dir + test_path, skiprows=[0,1,2,4,5])

    df_test.head()

    data_test = preprocess(df_test, time=0.25e6, remove_neutral=True)
    cols = data_test.columns

    print(all_attack_times)

    all_attack_indices = set()

    for start, end in all_attack_times:
        attack_indices = df_test.index[(df_test['Time'] >= start * 1e6) & (df_test['Time'] <= end * 1e6)] - 1201
        all_attack_indices.add((attack_indices[0], attack_indices[-1]))

    all_attack_indice = sorted(list(all_attack_indices))

    attacks_in = {}
    co = 0
    for key, value in attacks.items():
        attacks_in[sorted(all_attack_indice)[co][0]] = value
        co += 1

    # Eval

    corr_list = []
    var_list = []

    start_time = time.time()

    for var1_index, var2_indices in adj_list.items():

        var1_index -= 1
        var2_indices = np.array(var2_indices) - 1

        corr_list.append(
            stat_diff(data_test, var1_index, var2_indices, window=window, stat_fn=correlation_pair, fn="sum")
        )


    for var1_index, var2_indices in adj_list.items():

        var1_index -= 1
        var2_indices = np.array(var2_indices) - 1

        var_list.append(
            stat_diff(data_test, var1_index, var2_indices, window=window, stat_fn=var_pair, fn="sum")
        )


    end_time = time.time()


    rsum_list_corr = np.array(corr_list)
    rsum_list_var = np.array(var_list)

    time_taken_total = end_time - start_time
    avg_time = time_taken_total / (len(data_test) - window - w + 1)
    print(time_taken_total, avg_time)
    # in seconds

    total_normal = rsum_list_var.shape[1] + 1
    for start, end in all_attack_indices:
        total_normal -= (end - start + 1 + 2 * (window + w - 1))
    print(total_normal)

    thresholds_corr = np.array(thresholds_corr)
    thresholds_corr_99 = np.array(thresholds_corr_99)
    thresholds_corr_95 = np.array(thresholds_corr_95)

    thresholds_var = np.array(thresholds_var)
    thresholds_var_99 = np.array(thresholds_var_99)
    thresholds_var_95 = np.array(thresholds_var_95)

    quantiles = [1., 0.99, 0.95]

    rsum_list = rsum_list_corr

    print("Correlation")

    for i, threshold in enumerate([thresholds_corr, thresholds_corr_99, thresholds_corr_95]):
        print("Threshold:", quantiles[i])

        for persistency in [1, 3, 5]:
            print("Persistency:", persistency)


            # each row is sensor, each col is timestep
            violations = (rsum_list > threshold.reshape(-1, 1))
            ADD = detection_delay(violations.any(axis=0), sorted(list(all_attack_indices)), window, w, persistency)
            print("DD:", ADD)
            LD = localisation_delay(violations, sorted(list(all_attack_indices)), window, w, persistency)
            print("LD:", LD)
            fpr = FPR(violations.any(axis=0), all_attack_indices, window, w, total_normal, persistency)
            print("FPR", fpr)

            fpr_l = []
            for v in violations:
                fpr_l.append(FPR(v, all_attack_indices, window, w, total_normal, persistency=persistency))
            fpr_local = FPR_Local(violations, all_attack_indices, attacks_shifted_index, window, w, persistency)
            print("FPR Localisation (Detection):", np.mean(fpr_l), "+-", np.std(fpr_l))
            print("####################################################")
            print("FPR Localisation:", np.mean(fpr_local), "+-", np.std(fpr_local))
            print("FPR Localisation:", fpr_local)
            print("****************************************************")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print()


    rsum_list = rsum_list_var

    print("Variance")


    for i, threshold in enumerate([thresholds_var, thresholds_var_99, thresholds_var_95]):
        print("Threshold:", quantiles[i])

        for persistency in [1, 3, 5]:
            print("Persistency:", persistency)


            # each row is sensor, each col is timestep
            violations = (rsum_list > threshold.reshape(-1, 1))
            ADD = detection_delay(violations.any(axis=0), sorted(list(all_attack_indices)), window, w, persistency)
            print("DD:", ADD)
            LD = localisation_delay(violations, sorted(list(all_attack_indices)), window, w, persistency)
            print("LD:", LD)
            fpr = FPR(violations.any(axis=0), all_attack_indices, window, w, total_normal, persistency)
            print("FPR", fpr)

            fpr_l = []
            for v in violations:
                fpr_l.append(FPR(v, all_attack_indices, window, w, total_normal, persistency=persistency))
            fpr_local = FPR_Local(violations, all_attack_indices, attacks_shifted_index, window, w, persistency)
            print("FPR Localisation (Detection):", np.mean(fpr_l), "+-", np.std(fpr_l))
            print("####################################################")
            print("FPR Localisation:", np.mean(fpr_local), "+-", np.std(fpr_local))
            print("FPR Localisation:", fpr_local)
            print("****************************************************")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print()


    print("Combined")

    t_corr = [thresholds_corr, thresholds_corr_99, thresholds_corr_95]
    t_var = [thresholds_var, thresholds_var_99, thresholds_var_95]

    time_total = []

    for i in range(len(t_corr)):
        print("Threshold:", quantiles[i])

        for persistency in [1, 3, 5, 10]:
            print("Persistency:", persistency)


            # each row is sensor, each col is timestep
            violations = np.logical_or((rsum_list_corr > t_corr[i].reshape(-1, 1)), (rsum_list_var > t_var[i].reshape(-1, 1)))
            ADD = detection_delay(violations.any(axis=0), sorted(list(all_attack_indices)), window, w, persistency)
            print("DD:", ADD)
            ADD = [i if i is not None else np.nan for i in ADD]
            dd_mean = np.nanmean(ADD)
            print("Average DD:", dd_mean, np.nanstd(ADD))
            LD = localisation_delay(violations, sorted(list(all_attack_indices)), window, w, persistency)
            print("LD:", LD)
            lds = [ld if ld is not None else np.nan for sublist in LD for ld in sublist]
            ld_mean = np.nanmean(lds)
            print("Average LD:", ld_mean, "+-", np.nanstd(lds))
            fpr = FPR(violations.any(axis=0), all_attack_indices, window, w, total_normal, persistency)
            print("FPR", fpr)

            fpr_l = []
            for v in violations:
                fpr_l.append(FPR(v, all_attack_indices, window, w, total_normal, persistency=persistency))
            fpr_local = FPR_Local(violations, all_attack_indices, attacks_shifted_index, window, w, persistency)
            print("FPR Localisation (Detection):", np.mean(fpr_l), "+-", np.std(fpr_l))
            print("####################################################")
            print("FPR Localisation:", np.mean(fpr_local), "+-", np.std(fpr_local))
            print("FPR Localisation:", fpr_local)
            print("****************************************************")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print()

    #     Viz

    all_sensors =[]
    for i in attacks.values():
        for a in i:
            all_sensors += a

    attack_i = {}
    for key, items in attacks_in.items():
        attack_i[key] = []
        for i in items:
            attack_i[key] += i

    LD_corr = []
    at_neighbor = {}

    rsum_list = rsum_list_corr
    threshold = thresholds_corr_99
    t_list = t_corr
    persistency = 5

    for index, i in enumerate(rsum_list):
        print("***********************************************************************************************")
        attack_in = (i > threshold[index]).nonzero()

        violations = (rsum_list > threshold.reshape(-1, 1))

        print(f"Sensor {things[index]}: {cols[things[index] - 1]}")
        at_neighbor[things[index]] = attack_in
        plt.title(f"Sensor {things[index]}: {cols[things[index] - 1]}")
        plt.xlabel("(Windowed) Timesteps")
        plt.ylabel("Correlation Change Statistic")

        for t in t_list:
            plt.axhline(t[index], color='orange')

        plt.axhline(threshold[index], color='red')

        if things[index] in attack_dict:
            # if sensor is being attacked

            # ground truth
            attack_indices = df_test.index[
                                 (df_test['Time'] >= attack_dict[things[index]][0] * 1e6) &
                                 (df_test['Time'] <= attack_dict[things[index]][1] * 1e6)] - 1201

            for start, end in sorted(list(all_attack_indices)):

                # if this is the attack time of the sensor, plot solid red lines
                if attack_indices[0] == start:
                    DD = detection_delay(violations[index], [(start, end)], window, w, persistency)[0]
                    LD_corr.append(DD)
                    print(f"DD = {DD}")
                    start_w = start - window - w
                    end_w = end + window + w - 1
                    plt.axvline(start_w, color='red')
                    plt.axvline(end_w, color='red')

                # otherwise, plot pink lines
                else:
                    start_w = start - window - w
                    end_w = end + window + w - 1
                    plt.axvline(start_w, color='pink', alpha=0.75)
                    plt.axvline(end_w, color='pink', alpha=0.75)

                    start_w = start - window - w
                    end_w = end + window + w - 1
                    for starts, ate in attack_i.items():
                        if starts == start and things[index] in ate:
                            DD = detection_delay(violations[index], [(start, end)], window, w, persistency)[0]
                            LD_corr.append(DD)
                            print(f"DD = {DD}")
                            plt.axvline(start_w, color='orange', alpha=0.75)
                            plt.axvline(end_w, color='orange', alpha=0.75)

        else:
            for start, end in sorted(list(all_attack_indices)):
                start_w = start - window - w
                end_w = end + window + w - 1
                plt.axvline(start_w, color='pink', alpha=0.75)
                plt.axvline(end_w, color='pink', alpha=0.75)
                for starts, ate in attack_i.items():
                    if starts == start:
                        if things[index] in ate:
                            DD = detection_delay(violations[index], [(start, end)], window, w, persistency)[0]
                            LD_corr.append(DD)
                            print(f"DD = {DD}")
                            plt.axvline(start_w, color='orange', alpha=0.75)
                            plt.axvline(end_w, color='orange', alpha=0.75)

        plt.plot(i)
        plt.show()

    print(LD_corr)

    LD_var = []
    at_neighbor = {}

    rsum_list = rsum_list_var
    threshold = thresholds_var_99
    t_list = [thresholds_var, thresholds_var_99, thresholds_var_95]
    persistency = 5

    for index, i in enumerate(rsum_list):
        print("***********************************************************************************************")
        attack_in = (i > threshold[index]).nonzero()

        violations = (rsum_list > threshold.reshape(-1, 1))

        print(f"Sensor {things[index]}: {cols[things[index] - 1]}")
        at_neighbor[things[index]] = attack_in
        plt.title(f"Sensor {things[index]}: {cols[things[index] - 1]}")
        plt.xlabel("(Windowed) Timesteps")
        plt.ylabel("Variance Ratio Change Statistic")

        for t in t_list:
            plt.axhline(t[index], color='orange')

        plt.axhline(threshold[index], color='red')

        if things[index] in attack_dict:
            # if sensor is being attacked

            # ground truth
            attack_indices = df_test.index[
                                 (df_test['Time'] >= attack_dict[things[index]][0] * 1e6) &
                                 (df_test['Time'] <= attack_dict[things[index]][1] * 1e6)] - 1201

            for start, end in sorted(list(all_attack_indices)):

                # if this is the attack time of the sensor, plot solid red lines
                if attack_indices[0] == start:
                    DD = detection_delay(violations[index], [(start, end)], window, w, persistency)[0]
                    LD_var.append(DD)
                    print(f"DD = {DD}")
                    start_w = start - window - w
                    end_w = end + window + w - 1
                    plt.axvline(start_w, color='red')
                    plt.axvline(end_w, color='red')

                # otherwise, plot pink lines
                else:
                    start_w = start - window - w
                    end_w = end + window + w - 1
                    plt.axvline(start_w, color='pink', alpha=0.75)
                    plt.axvline(end_w, color='pink', alpha=0.75)

                    start_w = start - window - w
                    end_w = end + window + w - 1
                    for starts, ate in attack_i.items():
                        if starts == start and things[index] in ate:
                            DD = detection_delay(violations[index], [(start, end)], window, w, persistency)[0]
                            LD_var.append(DD)
                            print(f"DD = {DD}")
                            plt.axvline(start_w, color='orange', alpha=0.75)
                            plt.axvline(end_w, color='orange', alpha=0.75)

        else:
            for start, end in sorted(list(all_attack_indices)):
                start_w = start - window - w
                end_w = end + window + w - 1
                plt.axvline(start_w, color='pink', alpha=0.75)
                plt.axvline(end_w, color='pink', alpha=0.75)
                for starts, ate in attack_i.items():
                    if starts == start:
                        if things[index] in ate:
                            DD = detection_delay(violations[index], [(start, end)], window, w, persistency)[0]
                            LD_var.append(DD)
                            print(f"DD = {DD}")
                            plt.axvline(start_w, color='orange', alpha=0.75)
                            plt.axvline(end_w, color='orange', alpha=0.75)

        plt.plot(i)
        plt.show()

    print(LD_var)

    # FLR

    n = 0
    flr = []
    violation = np.logical_or((rsum_list_corr > t_corr[1].reshape(-1, 1)), (rsum_list_var > t_var[1].reshape(-1, 1)))
    attack_pred = violation.nonzero()

    for i, (start, end) in enumerate(sorted(all_attack_indice)):
        print("***********************************************************************************************")
        print(f'For the {order_attack[n]} attack starting at index: {start} and ending at index: {end}')
        start_w = start - window - w + 1
        end_w = end + window + w - 1
        # get all indices that are after the start of attack
        indices = np.logical_and((attack_pred[1] >= start_w), (attack_pred[1] <= end_w))
        attack_window = attack_pred[1][indices]
        attack_sensors = attack_pred[0][indices]

        sensor_list = attacks_shifted_index[attack_start_times[i]]
        sensors_attacked = [s for sensor_l in sensor_list for s in sensor_l]

        false_localised_sensors = 0
        false_localised_sensors_list = []

        for sensor in np.unique(attack_sensors):
            if sensor not in sensors_attacked:
                sensor_indices = (attack_sensors == sensor)
                sensor_attack_window = np.sort(attack_window[sensor_indices])

                # check for persistency
                idx = persistency_check(sensor_attack_window, persistency)
                if idx is not None:
                    false_localised_sensors += 1
                    false_localised_sensors_list.append(sensor)

        print(f'1 Hop: {sensors_attacked}')
        print(f'False Localisations: {false_localised_sensors_list}')

        flr.append(false_localised_sensors / len(sensors_attacked))
        n += 1
    print(flr)
    print(np.mean(flr), np.std(flr))

    save_results(thresholds_corr_99, thresholds_var_99, rsum_list_corr, rsum_list_var, all_attack_indices,
                     all_attack_indice, total_normal, persistency=5, window=window, w=w,
                     save_path=f"models/results_{topology}.json")


if __name__ == '__main__':
    main()