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

function cgns_plot_part_kinetic_energy(casename, ts, te)
% CGNS_PLOT_PART_VELOCITY  Read the particle velocities from a BLUEBOTTLE-
%   generated CGNS file.
%
%   CGNS_PLOT_PART_VELOCITY(CASENAME, SIGFIGS) plots the particle velocites
%   from the simulation CASENAME between ts-te. Each of the velocity
%   components is an array representing all of the particles in the simulation.
%
%   Example:
%     cgns_plot_part_velocity('simulation',0,1) will plot the velocity history
%     for the CASENAME = 'simulation' between t = 0 and 1 

% read part time
[tstr tnum] = cgns_read_part_time(casename);
n = 1:1:length(tnum);

% 'cut' the time array between the ts and te values, inclusive
i = find(tnum<ts | tnum>te);
n(i) = [];
tnum(i) = [];
ts = tstr{n(1)};

% find the number of particles using the initial timestep
[x, y, z] = cgns_read_part_vel(casename, ts);

nt = length(n); % nt = number of timesteps
[np d] = size(x); % np = number of particles

% create data storage locations
T = zeros(nt, 1);
up = zeros(np, nt);
vp = zeros(np, nt);
wp = zeros(np, nt);

% read particle position data
for i = 1:nt
  [up(:,i), vp(:,i), wp(:,i)] = cgns_read_part_vel(casename, tstr{n(i)});

  um = mean(up,1);
  vm = mean(vp,1);
  wm = mean(wp,1);
  %us(:,i) = (up(:,i)-um(i)).*(up(:,i)-um(i));
  %vs(:,i) = (vp(:,i)-vm(i)).*(vp(:,i)-vm(i));
  %ws(:,i) = (wp(:,i)-wm(i)).*(wp(:,i)-wm(i));
  us(:,i) = up(:,i).*up(:,i);
  vs(:,i) = vp(:,i).*vp(:,i);
  ws(:,i) = wp(:,i).*wp(:,i);
end

k = mean(us+vs+ws,1);

figure
plot(tnum, k, 'ko')
xlabel('$t$', 'interpreter','latex')
ylabel('$k$', 'interpreter', 'latex')


