function cvfem2d_vis_global(fTimes,filename)
docNode = com.mathworks.xml.XMLUtils.createDocument('VTKFile');
vtk = docNode.getDocumentElement;
vtk.setAttribute('type','Collection');
vtk.setAttribute('version','1.0');
vtk.setAttribute('byte_order','LittleEndian');

collection = docNode.createElement('Collection');
vtk.appendChild(collection);

for i = 1 : length(fTimes)
    thisElement = docNode.createElement('DataSet');
    thisElement.setAttribute('file',sprintf('%s%s%s%08d.vtu',filename,filesep,filename,i));
    thisElement.setAttribute('group','');
    thisElement.setAttribute('part','0');
    thisElement.setAttribute('timestep',sprintf('%.12f',fTimes(i)));
    collection.appendChild(thisElement);
end
xmlwrite(sprintf('%s.pvd',filename),docNode);