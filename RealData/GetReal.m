clear all
close all

for flag=1:4
    if flag == 1
        data = dlmread("control.data", '\t');
        save_file='Real1';
        Pi=101325;
        save_file2='Inlet1';
    elseif flag==2
        data = dlmread("circular.data", '\t');
        save_file='Real2';
        save_file2='Inlet2';
        Pi=101325;
    elseif flag==3
        data = dlmread("RT.data", '\t');
        save_file='Real3';
        save_file2='Inlet3';
        Pi=101325;
    elseif flag==4
        data = dlmread("circular_RT.data", '\t');
        save_file='Real4';
        save_file2='Inlet4';
        Pi=101325;
    end% offset_index = 1;


    ind = find( (data(:,2)* Pi) > 50000, 1); % index of the first data point with p_inlet > 0.02 bar
    offset_time = data(ind, 1); % Time assumed to be the start of the injection

    duration = data(end,1)-offset_time; % 170 seconds
    end_ind = find( data(:,1) == duration +offset_time );

    p_ref_2 = data(1, 2);
    p_inlet = [(data(ind:end_ind,1) - offset_time) ((data(ind:end_ind,2)) * Pi)];


    time = [data(ind:end_ind, 1)];
    ob_times = time - offset_time;

    for i = 2:size(data,2)
        mean_noise = mean(data(1:20, i));
        data(:,i) = data(:, i) - mean_noise;
    end

    press_full = (data(ind:end_ind,3:end)  )* Pi;
    inlet = (data(ind:end_ind,2)  )* Pi;
    target_xy = [
        0.00, 0.15;
        0.0375, 0.005;
        0.1125, 0.005;
        0.1875, 0.005;
        0.2625, 0.005;
        0.05,   0.05;
        0.15,   0.05;
        0.25,   0.05;
        0.075,  0.075;
        0.225,  0.075;
        0.05,   0.15;
        0.15,   0.15;
        0.25,   0.15;
        0.075,  0.225;
        0.225,  0.225;
        0.05,   0.25;
        0.15,   0.25;
        0.25,   0.25;
        0.025,  0.295;
        0.075,  0.295;
        0.125,  0.295;
        0.175,  0.295;
        0.225,  0.295;
        0.275,  0.295
        ];
    press_full_corr = press_full;

    save(save_file,"press_full_corr","ob_times")
    save(save_file2,"inlet","ob_times")
end
