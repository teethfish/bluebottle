%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BLUEBOTTLE-1.0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Copyright 2012 - 2014 Adam Sierakowski, The Johns Hopkins University
% 
%   Licensed under the Apache License, Version 2.0 (the "License");
%   you may not use this file except in compliance with the License.
%   You may obtain a copy of the License at
% 
%       http://www.apache.org/licenses/LICENSE-2.0
% 
%   Unless required by applicable law or agreed to in writing, software
%   distributed under the License is distributed on an "AS IS" BASIS,
%   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%   See the License for the specific language governing permissions and
%   limitations under the License.
% 
%   Please contact the Johns Hopkins University to use Bluebottle for
%   commercial and/or for-profit applications.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [u, v, w, t] = cgns_read_point_vel(casename, time)
% CGNS_READ_POINT_VEL  Read the flow velocity time series from a BLUEBOTTLE-
%   generated CGNS file at a point.
%
%   CGNS_READ_POINT_VEL(CASENAME, TS, TE, X, Y, Z) plots the particle velocity
%   time series at point (X, Y, Z) from the simulation CASENAME between TS
%   and TE.
%
%   Example:
%     cgns_read_point_vel('simulation',0,1,2,3,4) will plot the velocity time
%     series for the CASENAME = 'simulation' between t = 0 and 1 at
%     point (2,3,4).


if isa(time, 'double') == 1
    % find the directory contents
    path = [casename '/output'];
    od = cd(path);
    contents = dir;
    % Get names of files, remove first two (.., .), find flows/parts
    names = {contents.name}';
    names(1:2) = [];
    check = find(strncmp(names, 'point', 5));
    % Get sigfigs
    sigfigs = names{check(1)};
    sigfigs = length(sigfigs(7:end-5)) - 2;
    t_format = sprintf('%%1.%df', sigfigs);
    tt = sprintf(t_format, time);
    cd(od);
elseif isa(time, 'char') == 1
    tt = time;
end

path = [casename '/output/point-' tt '.cgns'];

usol = '/Base/Zone0/Solution/VelocityX/ data';
vsol = '/Base/Zone0/Solution/VelocityY/ data';
wsol = '/Base/Zone0/Solution/VelocityZ/ data';

u = h5read(path, usol);
v = h5read(path, vsol);
w = h5read(path, wsol);

