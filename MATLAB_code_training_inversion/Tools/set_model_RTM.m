function Model = set_model_RTM(Dx, Nx, dataTag)
%SET_MODEL_RTM  Build RTM forward model and sensor layout.
%
%   Model = set_model_RTM(Dx, Nx, dataTag)
%
%   Dx      : domain length in x-direction (meters)
%   Nx      : number of grid nodes along each side (discretization)
%   dataTag : string flag to choose sensor layout / data type:
%             '_for_real' -> lab sensor positions (real sensors)
%             '_100'      -> synthetic 10x10 grid of 100 sensors
%
%   If dataTag is omitted, it defaults to '_for_real'.

    if nargin < 3 || isempty(dataTag)
        dataTag = '_for_real';  % default
    end

    % interpret tag
    useLabSensors = strcmpi(dataTag, '_for_real');  % lab vs 100-grid

    addpath('Tools/MATCVFEM-master', 'Tools/MATFEM-master');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Geometry for RTM domain (square [0,Dx] × [0,Dx])
    Dy = Dx;

    lowerLeft  = [0,   0];
    lowerRight = [Dx,  0];
    upperRight = [Dx,  Dy];
    upperLeft  = [0,  Dy];

    S = [3, 4, ...
         lowerLeft(1),  lowerRight(1), upperRight(1), upperLeft(1), ...
         lowerLeft(2),  lowerRight(2), upperRight(2), upperLeft(2)];

    gdm = S';
    ns  = 'S';
    sf  = 'S';

    g   = decsg(gdm, ns, sf');

    RTM = createpde;
    geometryFromEdges(RTM, g);

    %% Mesh resolution
    generateMesh(RTM,'Hmax',0.006,'GeometricOrder','linear');

    [p, e, t] = meshToPet(RTM.Mesh);
    mesh = convert_pdemesh(p, e, t);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Flow properties

    darcy.thickness = 0.001;
    darcy.mu        = 0.00;
    darcy.phi       = 0.0;

    vis.dumpIdx     = 1;
    vis.dumpFlag    = false;
    vis.filename    = 'recDomain';

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Boundary conditions

    bndry.gd_filename   = 'g_D2';
    bndry.gn_filename   = 'g_N';
    bndry.pvent         = 0;
    bndry.pinlet        = 110000;

    bndry.inlet_location_fname = 'rec_inlet_location';
    bndry.vent_location_fname  = 'rec_vent_location';

    %% Setup CVFEM solver
    K = 1e-9 * ones(size(mesh.elem, 1), 1);
    opt = cvfem_setup(mesh, bndry, vis, darcy, K);
    opt = cvfem2d_init(opt);
    Model.opt = opt;

    Nodes = opt.mesh.node';
    save("MATLAB_files_for_emulator/Nodes.mat","Nodes");

    %% Mass matrix
    MassMatrix = assembling_mass_tri(opt.mesh.node, opt.mesh.elem);
    save('MATLAB_files_for_emulator/MassMatrix.mat','MassMatrix','-v7');

    %% Observation times
    Model.ob_times = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ...
                      17 19 21 23 25 27 30 35 40 45 50 55 ...
                      60 65 70 80 90 100 110];

    %% Store general model information
    Model.Dx      = Dx;
    Model.Dy      = Dy;
    Model.Nx      = Nx;
    Model.RTM     = RTM;
    Model.dataTag = dataTag;   % store which tag was used

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Sensor Layout (depends on dataTag)
    if useLabSensors
        %% --- LAB SENSOR POSITION SET (REAL / '_for_real') ---
        target_xy = [
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

        % Node coordinates from PDE mesh
        nodes = RTM.Mesh.Nodes';  

        tol     = 1e-6;
        indices = zeros(size(target_xy, 1), 1);

        for k = 1:size(target_xy, 1)
            dist = vecnorm(nodes - target_xy(k, :), 2, 2);
            [minVal, idx] = min(dist);

            if minVal > tol
                warning('No close match to (%.4f, %.4f)', ...
                    target_xy(k,1), target_xy(k,2));
            end

            indices(k) = idx;
        end

        Model.nodes_meas  = indices;
        Model.sensor_type = "lab";
        ind = indices - 1;

        % use dataTag in filename if you like, or keep old name
        save("MATLAB_files_for_emulator/indices_for_real.mat","ind");
        save('MATLAB_files_for_emulator/save_nodes_for_plotting_real', 'target_xy');

    else
        %% --- SYNTHETIC 10×10 GRID OF 100 SENSORS ('_100') ---
        M = 10;

        [xx, yy] = meshgrid(linspace(0.005, Dx - 0.005, M));
        x        = xx(:);
        y        = yy(:);

        nodes_meas = zeros(M*M, 1);
        for k = 1:(M*M)
            nodes_meas(k) = dsearchn(opt.mesh.node, [x(k), y(k)]);
        end

        Model.nodes_meas  = nodes_meas;
        Model.sensor_type = "grid100";
        ind = nodes_meas - 1;

        save("MATLAB_files_for_emulator/indices_100.mat","ind");
        grid_xy = [x, y];
        save('MATLAB_files_for_emulator/save_nodes_for_plotting_100', 'grid_xy', 'xx', 'yy');
    end

end
