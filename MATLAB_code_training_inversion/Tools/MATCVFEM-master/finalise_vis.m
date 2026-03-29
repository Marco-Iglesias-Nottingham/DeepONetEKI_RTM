function finalise_vis(opt)
opt.vis.dumpIdx = opt.vis.dumpIdx - 1;

[flag, ME,MID] = mkdir(opt.vis.filename);
if strcmp(MID,'MATLAB:MKDIR:DirectoryExists')
    system(['rm -rf ' opt.vis.filename]);
    mkdir(opt.vis.filename);
end

cwd = pwd;
movefile('*.vtu',[cwd filesep opt.vis.filename])
cvfem2d_vis_global(opt.vis.fTimes(1:opt.vis.dumpIdx),opt.vis.filename)