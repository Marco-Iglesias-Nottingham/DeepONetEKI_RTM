function cvfem2d_vis_permeability(opt)
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
if opt.cvfem.ktype == 1
    piece.setAttribute('NumberOfPoints',num2str(opt.mesh.nnode));
    piece.setAttribute('NumberOfCells',num2str(opt.mesh.nelem));
else
    piece.setAttribute('NumberOfPoints',num2str(opt.mesh.nelem));
    piece.setAttribute('NumberOfCells',num2str(opt.mesh.nelem));
end
ugrid.appendChild(piece);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% VTKFile>UnstructuredGrid>Piece>PointData
%
scalar_pdata = docNode.createElement('PointData');
scalar_pdata.setAttribute('Scalars','scalars');
piece.appendChild(scalar_pdata);

if opt.cvfem.ktype == 3
    k = docNode.createElement('DataArray');
    k.setAttribute('Format','ascii')
    k.setAttribute('Name','Permeability')
    k.setAttribute('NumberOfComponents','9')
    k.setAttribute('type','Float64')
    k.appendChild( docNode.createTextNode(sprintf('\n')));
    k.appendChild(docNode.createTextNode(sprintf('%12.6e %12.6e 0 %12.6e %12.6e 0 0 0 0\n',opt.cvfem.K(:,[1 3 3 2])')));
    k.appendChild( docNode.createTextNode(sprintf('\n')));
    scalar_pdata.appendChild(k);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%
% VTKFile>UnstructuredGrid>Piece>CellData
%
scalar_cdata = docNode.createElement('CellData');
scalar_cdata.setAttribute('Scalars','scalars');
piece.appendChild(scalar_cdata);


if opt.cvfem.ktype == 1
    k = docNode.createElement('DataArray');
    k.setAttribute('Format','ascii')
    k.setAttribute('Name','log permeability')
    k.setAttribute('type','Float64')
    k.appendChild( docNode.createTextNode(sprintf('\n')));
    k.appendChild(docNode.createTextNode(sprintf('%12.6e ',log(opt.cvfem.K(:,1)))));
    k.appendChild( docNode.createTextNode(sprintf('\n')));
    scalar_cdata.appendChild(k);
end


if opt.cvfem.ktype == 1
    k2 = docNode.createElement('DataArray');
    k2.setAttribute('Format','ascii')
    k2.setAttribute('Name','variance')
    k2.setAttribute('type','Float64')
    k2.appendChild( docNode.createTextNode(sprintf('\n')));
    k2.appendChild(docNode.createTextNode(sprintf('%12.6e ',log(opt.cvfem.VAR))));
    k2.appendChild( docNode.createTextNode(sprintf('\n')));
    scalar_cdata.appendChild(k2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if opt.cvfem.ktype == 1
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
    
else
    centre = (opt.mesh.node(opt.mesh.elem(:,1),:)+opt.mesh.node(opt.mesh.elem(:,2),:)+opt.mesh.node(opt.mesh.elem(:,3),:))/3;
   
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
    pdata_array.appendChild(docNode.createTextNode(sprintf('%f %f %f\n',[centre zeros(opt.mesh.nelem,1)]')));
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
    connectivity.appendChild(docNode.createTextNode(sprintf('%d\n',0:opt.mesh.nelem-1)));
    cells.appendChild(connectivity);
    
    %
    % VTKFile>UnstructuredGrid>Piece>Cells>DataArray(offsets)
    %
    offsets = docNode.createElement('DataArray');
    offsets.setAttribute('Format','ascii');
    offsets.setAttribute('Name','offsets');
    offsets.setAttribute('type','Int64');
    offsets.appendChild( docNode.createTextNode(sprintf('\n')));
    offsets.appendChild(docNode.createTextNode(sprintf('%d ',1:opt.mesh.nelem)));
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
    types.appendChild(docNode.createTextNode(sprintf('%d ',ones(opt.mesh.nelem,1))));
    types.appendChild(docNode.createTextNode(sprintf('\n')));
    cells.appendChild(types);
end
xmlwrite(sprintf('%s_permeability.vtu',opt.vis.filename),docNode);
