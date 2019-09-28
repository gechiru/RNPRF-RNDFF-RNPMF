%     demo for probability fusion
%
%--------------Brief description-------------------------------------------
%
% 
% This demo implements probability fusion for hyperspectral and LiDAR fusion classification[1]
%
%
% More details in [1]:
%
% [1]C. Ge, Q. Du, W. Sun, K. Wang, J. Li, and Y. Li, "Hyperspectral and LiDAR Data Fusion Classification 
%    Using Residual Network-based Deep Feature Fusion and Probability Fusion,"  IEEE Transactions on Geoscience 
%    and Remote Sensing, 2019.
%
% contact: gechiru@126.com (Chiru Ge)

clear
clc
%% show HSI map
load('HSImap.mat')
map_HSI=[];
for i=1:11
map_HSI=[map_HSI,imagemap{i}];
end
map_HSI=map_HSI+1;

Probability_HSI=[];
for j=1:11
Probability_HSI=[Probability_HSI;imageprob{j}];
end
Probability_HSI=Probability_HSI;

map=reshape(map_HSI,1905,349);
MyColorMap = [0,0,1; 0,1,0; 0,1,1; 1,0,0; 1,0,1; 1,1,0;...
              0.5,0.5,0.5; 0.5,0.5,1; 0.5,1,0.5; 1,0.5,0.5; 0.5,1,1; 1,0.5,1; 1,1,0.5; 0.3,0.7,0.3;0,0.2,0.5];%;0.8,0.8,0.8
figure;imagesc(map'); set(gca,'xtick',[],'ytick',[]); colormap(MyColorMap);

%% show HSI_EPLBP map
load('HSI_EPLBPmap.mat')
map_HSI=[];
for i=1:11
map_HSI=[map_HSI,imagemap{i}];
end
map_HSI=map_HSI+1;

Probability_HSI=[];
for j=1:11
Probability_HSI=[Probability_HSI;imageprob{j}];
end
Probability_HSI=Probability_HSI;
map=reshape(map_HSI,1905,349);
MyColorMap = [0,0,1; 0,1,0; 0,1,1; 1,0,0; 1,0,1; 1,1,0;...
              0.5,0.5,0.5; 0.5,0.5,1; 0.5,1,0.5; 1,0.5,0.5; 0.5,1,1; 1,0.5,1; 1,1,0.5; 0.3,0.7,0.3;0,0.2,0.5];%;0.8,0.8,0.8
figure;imagesc(map'); set(gca,'xtick',[],'ytick',[]); colormap(MyColorMap);

%% show LiDAR_EPLBP map
load('LiDAR_EPLBPmap.mat')
map_HSI=[];
for i=1:11
map_HSI=[map_HSI,imagemap{i}];
end
map_HSI=map_HSI+1;

Probability_HSI=[];
for j=1:11
Probability_HSI=[Probability_HSI;imageprob{j}];
end
Probability_HSI=Probability_HSI;
map=reshape(map_HSI,1905,349);
MyColorMap = [0,0,1; 0,1,0; 0,1,1; 1,0,0; 1,0,1; 1,1,0;...
              0.5,0.5,0.5; 0.5,0.5,1; 0.5,1,0.5; 1,0.5,0.5; 0.5,1,1; 1,0.5,1; 1,1,0.5; 0.3,0.7,0.3;0,0.2,0.5];%;0.8,0.8,0.8
figure;imagesc(map'); set(gca,'xtick',[],'ytick',[]); colormap(MyColorMap);

%% show RNDFF map
load('3FF_map.mat')
map_HSI=[];
for i=1:61
map_HSI=[map_HSI,imagemap{i}];
end
map_HSI=map_HSI+1;

Probability_HSI=[];
for j=1:61
Probability_HSI=[Probability_HSI;imageprob{j}];
end
Probability_HSI=Probability_HSI;
map=reshape(map_HSI,1905,349);
MyColorMap = [0,0,1; 0,1,0; 0,1,1; 1,0,0; 1,0,1; 1,1,0;...
              0.5,0.5,0.5; 0.5,0.5,1; 0.5,1,0.5; 1,0.5,0.5; 0.5,1,1; 1,0.5,1; 1,1,0.5; 0.3,0.7,0.3;0,0.2,0.5];%;0.8,0.8,0.8
figure;imagesc(map'); set(gca,'xtick',[],'ytick',[]); colormap(MyColorMap);

%% residual network-based probability reconstruction fusion (RNPRF) map and the corresponding combination method
load('allmap1.mat')
load('allmap2.mat')
a=0.61;
b=0.72;
tmp1=a*b.*Probability_HSI; 
tmp2=b*(1-a).*Probability_HSI_EPLBP;
tmp3=(1-b).*Probability_LiDAR_EPLBP;
tmp=tmp1+tmp2+tmp3; % RNPRF
[value class] = max(tmp');

tmp1=0.01.*tmp+0.99.*Probability_3FF;  %RNPRF+RNDFF
[value class] = max(tmp1');

tmp1=tmp.*Probability_3FF; % RNPRF.*RNDFF
[value class] = max(tmp1');

tmp1=Probability_HSI.*Probability_HSI_EPLBP.*Probability_LiDAR_EPLBP.*Probability_3FF; % RNPMF.*RNDFF
[value class] = max(tmp1');

map=reshape(class,1905,349);
MyColorMap = [0,0,1; 0,1,0; 0,1,1; 1,0,0; 1,0,1; 1,1,0;...
              0.5,0.5,0.5; 0.5,0.5,1; 0.5,1,0.5; 1,0.5,0.5; 0.5,1,1; 1,0.5,1; 1,1,0.5; 0.3,0.7,0.3;0,0.2,0.5];%;0.8,0.8,0.8
figure;imagesc(map'); set(gca,'xtick',[],'ytick',[]); colormap(MyColorMap);

%% residual network-based probability multiplication fusion (RNPMF)
load('allmap1.mat')
load('allmap2.mat')
tmp=(Probability_HSI).*(Probability_HSI_EPLBP).*(Probability_LiDAR_EPLBP); 
[value class] = max(tmp');
map=reshape(class,1905,349);
MyColorMap = [0,0,1; 0,1,0; 0,1,1; 1,0,0; 1,0,1; 1,1,0;...
              0.5,0.5,0.5; 0.5,0.5,1; 0.5,1,0.5; 1,0.5,0.5; 0.5,1,1; 1,0.5,1; 1,1,0.5; 0.3,0.7,0.3;0,0.2,0.5];%;0.8,0.8,0.8
figure;imagesc(map'); set(gca,'xtick',[],'ytick',[]); colormap(MyColorMap);

%% max voting map
%If the results of the three classifiers are different, they are based on HSI, HSI_EPLBP, and LiDAR_EPLBP respectively.
voting=[map_HSI;map_HSI_EPLBP;map_LiDAR_EPLBP];
result=zeros(1,length(voting));
%HSI
for i=1:1:length(voting)
    if voting(2,i)==voting(3,i)& voting(1,i)~=voting(2,i)
    result(i)=voting(2,i);
    else
    result(i)=voting(1,i);
    end
end
%HSI_EP
for i=1:1:length(voting)
    if voting(1,i)==voting(3,i)& voting(1,i)~=voting(2,i)
    result(i)=voting(1,i);
    else
    result(i)=voting(2,i);
    end
end
%LiDAR_EP
for i=1:1:length(voting)
    if voting(1,i)==voting(2,i)&& voting(1,i)~=voting(3,i)
    result(i)=voting(2,i);
    else
    result(i)=voting(3,i);
    end
end

map=reshape(result,1905,349);
MyColorMap = [0,0,1; 0,1,0; 0,1,1; 1,0,0; 1,0,1; 1,1,0;...
              0.5,0.5,0.5; 0.5,0.5,1; 0.5,1,0.5; 1,0.5,0.5; 0.5,1,1; 1,0.5,1; 1,1,0.5; 0.3,0.7,0.3;0,0.2,0.5];%;0.8,0.8,0.8
figure;imagesc(map'); set(gca,'xtick',[],'ytick',[]); colormap(MyColorMap);
