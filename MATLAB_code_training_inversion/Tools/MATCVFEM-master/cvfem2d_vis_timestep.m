function cvfem2d_vis_timestep(opt)
opt.cvfem.u(isnan(opt.cvfem.u))=-1;
%
% VTKFile
%
docNode = com.mathworks.xml.XMLUtils.createDocument('VTKFile');
vtk = docNode.getDocumentElement;
% vtk.setAttribute('header_type','UInt64')
vtk.setAttribute('type','UnstructuredGrid');
vtk.setAttribute('version','1.0');
vtk.setAttribute('byte_order','BigEndian');

%
% VTKFile>UnstructuredGrid
%
ugrid = docNode.createElement('UnstructuredGrid');
vtk.appendChild(ugrid);

%
% VTKFile>UnstructuredGrid>Piece
%
piece = docNode.createElement('Piece');
piece.setAttribute('NumberOfPoints',num2str(opt.mesh.nnode));
piece.setAttribute('NumberOfCells',num2str(opt.mesh.nelem));
ugrid.appendChild(piece);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% VTKFile>UnstructuredGrid>Piece>PointData
%
scalar_pdata = docNode.createElement('PointData');
scalar_pdata.setAttribute('Scalars','scalars');
piece.appendChild(scalar_pdata);

%
% VTKFile>UnstructuredGrid>Piece>PointData>DataArray(pressure)
%
pressure = docNode.createElement('DataArray');
pressure.setAttribute('Format','ascii');
pressure.setAttribute('Name','Pressure');
pressure.setAttribute('type','Float64');
pressure.appendChild( docNode.createTextNode(sprintf('\n')));
pressure.appendChild(docNode.createTextNode(sprintf('%f ',max(opt.cvfem.u,opt.bndry.pvent))));
%pressure.appendChild(docNode.createTextNode(sprintf('%f ',opt.cvfem.u)));
pressure.appendChild( docNode.createTextNode(sprintf('\n')));
scalar_pdata.appendChild(pressure);

%
% VTKFile>UnstructuredGrid>Piece>PointData>DataArray(boundary)
%
bnd = opt.bndry.Dirichlet;
bnd(opt.bndry.neumann_flag ==1 & opt.cvfem.activeNode==1) = 2;
boundary = docNode.createElement('DataArray');
boundary.setAttribute('Format','ascii');
boundary.setAttribute('Name','Boundary');
boundary.setAttribute('type','Int64');
boundary.appendChild( docNode.createTextNode(sprintf('\n')));
boundary.appendChild(docNode.createTextNode(sprintf('%d ',bnd)));
boundary.appendChild( docNode.createTextNode(sprintf('\n')));
scalar_pdata.appendChild(boundary);

%
% VTKFile>UnstructuredGrid>Piece>PointData>DataArray(fFactor)
%
fFactor = docNode.createElement('DataArray');
fFactor.setAttribute('Format','ascii');
%fFactor.setAttribute('Name','Filling_Factor');
fFactor.setAttribute('Name','Front_Location');
fFactor.setAttribute('type','Float64');
fFactor.appendChild( docNode.createTextNode(sprintf('\n')));
%fFactor.appendChild(docNode.createTextNode(sprintf('%f ',opt.cvfem.fFactor)));
fFactor.appendChild(docNode.createTextNode(sprintf('%f ',opt.cvfem.fFactor)));
fFactor.appendChild( docNode.createTextNode(sprintf('\n')));
scalar_pdata.appendChild(fFactor);

%
% VTKFile>UnstructuredGrid>Piece>PointData>DataArray(voidID)
%
% vID = docNode.createElement('DataArray');
% vID.setAttribute('Format','ascii');
% %vID.setAttribute('Name','VoidID');
% vID.setAttribute('Name','log_perm');
% vID.setAttribute('type','Float64');
% vID.appendChild( docNode.createTextNode(sprintf('\n')));
% %vID.appendChild(docNode.createTextNode(sprintf('%f ',opt.cvfem.voidID)));
% vID.appendChild(docNode.createTextNode(sprintf('%12.6e ',log(opt.cvfem.K(:,1)))));
% vID.appendChild( docNode.createTextNode(sprintf('\n')));
% scalar_cpdata.appendChild(vID);
% %




% VTKFile>UnstructuredGrid>Piece>PointData>DataArray(activeNode)
%
activeNode = docNode.createElement('DataArray');
activeNode.setAttribute('Format','ascii');
activeNode.setAttribute('Name','Active_Node');
activeNode.setAttribute('type','Float64');
activeNode.appendChild( docNode.createTextNode(sprintf('\n')));
activeNode.appendChild(docNode.createTextNode(sprintf('%f ',opt.cvfem.activeNode)));
activeNode.appendChild( docNode.createTextNode(sprintf('\n')));
scalar_pdata.appendChild(activeNode);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%
% VTKFile>UnstructuredGrid>Piece>CellData
%
scalar_cdata = docNode.createElement('CellData');
scalar_cdata.setAttribute('Scalars','scalars');
piece.appendChild(scalar_cdata);

activeElement = docNode.createElement('DataArray');
activeElement.setAttribute('Format','ascii')
activeElement.setAttribute('Name','Active_Element')
activeElement.setAttribute('type','Float64')
activeElement.appendChild( docNode.createTextNode(sprintf('\n')));
activeElement.appendChild(docNode.createTextNode(sprintf('%6.2f ',opt.cvfem.activeElement)));
activeElement.appendChild( docNode.createTextNode(sprintf('\n')));
scalar_cdata.appendChild(activeElement);


k = docNode.createElement('DataArray');
k.setAttribute('Format','ascii')
k.setAttribute('Name','log permeability')
k.setAttribute('type','Float64')
k.appendChild( docNode.createTextNode(sprintf('\n')));
k.appendChild(docNode.createTextNode(sprintf('%12.6e ',log(opt.cvfem.K(:,1)))));
k.appendChild( docNode.createTextNode(sprintf('\n')));
scalar_cdata.appendChild(k);

k2 = docNode.createElement('DataArray');
k2.setAttribute('Format','ascii')
k2.setAttribute('Name','Porosity')
k2.setAttribute('type','Float64')
k2.appendChild( docNode.createTextNode(sprintf('\n')));
% k2.appendChild(docNode.createTextNode(sprintf('%12.6e ',log(opt.cvfem.K(:,1)))));
k2.appendChild(docNode.createTextNode(sprintf('%12.6e ',opt.darcy.phi)));
k2.appendChild( docNode.createTextNode(sprintf('\n')));
scalar_cdata.appendChild(k2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
% VTKFile>UnstructuredGrid>Piece>Points
%
points = docNode.createElement('Points');
piece.appendChild(points);

%
% VTKFile>UnstructuredGrid>Piece>Points>DataArray(NumberOfComponents)
%
pdata_array = docNode.createElement('DataArray');
pdata_array.setAttribute('Format','ascii');
pdata_array.setAttribute('NumberOfComponents','3');
pdata_array.setAttribute('type','Float64');
pdata_array.appendChild( docNode.createTextNode(sprintf('\n')));
pdata_array.appendChild(docNode.createTextNode(sprintf('%f %f %f\n',[opt.mesh.node zeros(opt.mesh.nnode,1)]')));
points.appendChild(pdata_array);

%
% VTKFile>UnstructuredGrid>Piece>Cells
%
cells = docNode.createElement('Cells');
piece.appendChild(cells);

%
% VTKFile>UnstructuredGrid>Piece>Cells>DataArray(connectivity)
%
connectivity = docNode.createElement('DataArray');
connectivity.setAttribute('Format','ascii');
connectivity.setAttribute('Name','connectivity');
connectivity.setAttribute('type','Int64');
connectivity.appendChild( docNode.createTextNode(sprintf('\n')));
connectivity.appendChild(docNode.createTextNode(sprintf('%d %d %d\n',opt.mesh.elem'-1)));
cells.appendChild(connectivity);

%
% VTKFile>UnstructuredGrid>Piece>Cells>DataArray(offsets)
%
offsets = docNode.createElement('DataArray');
offsets.setAttribute('Format','ascii');
offsets.setAttribute('Name','offsets');
offsets.setAttribute('type','Int64');
offsets.appendChild( docNode.createTextNode(sprintf('\n')));
offsets.appendChild(docNode.createTextNode(sprintf('%d ',3:3:3*opt.mesh.nelem)));
offsets.appendChild( docNode.createTextNode(sprintf('\n')));
cells.appendChild(offsets);

%
% VTKFile>UnstructuredGrid>Piece>Cells>DataArray(types)
%
types =  docNode.createElement('DataArray');
types.setAttribute('Format','ascii');
types.setAttribute('Name','types');
types.setAttribute('type','Int64');
types.appendChild( docNode.createTextNode(sprintf('\n')));
types.appendChild(docNode.createTextNode(sprintf('%d ',opt.mesh.elem_type)));
types.appendChild(docNode.createTextNode(sprintf('\n')));
cells.appendChild(types);


xmlwrite(sprintf('%s%08d.vtu',opt.vis.filename,opt.vis.dumpIdx),docNode);
