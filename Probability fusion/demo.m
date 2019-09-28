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

clc
clear
%% load data
load('testlabel.mat')
load('Houston_LiDAREPLBP_3-2_32_11x11_0.0003.mat')
load('Houston_HSIEPLBP_3-2_32_11x11_0.0003.mat')
load('Houston_HSI_2-2_16_11x11_0.0003.mat')
load('Houston_3FF_7-7_2-2_24_0.0003.mat')
[Y,I] = max(testlabel,[],2);
testlabels=I;
clear Y I

%% residual network-based probability multiplication fusion (RNPMF)
OA_max=0;
CA=[];AA=[];KA=[];
tmp=(Probability_HSI).*(Probability_HSI_EPLBP).*(Probability_LiDAR_EPLBP); 
[value class] = max(tmp');
S=[];
for i=1:15
[N]=find(testlabels==i);
S=[S;length(N)];
end

[confusion, accuracy_CRT, TPR, AA1, KA1, FPR] = confusion_matrix_GCR(class, S);
OA_max=accuracy_CRT;
CA=TPR;
AA=AA1;
KA=KA1;
A=[AA;OA_max;KA;CA'];

%% residual network-based probability reconstruction fusion (RNPRF)
OA_max=0;
CA=[];AA=[];KA=[];
S=[];
classLabel=[1:15];
for i=1:15
[N]=find(testlabels==i);
S=[S;length(N)];
end

load('c.mat')% the location of the validation set samples
testlabels1=testlabels(c);
Probability_HSI1=Probability_HSI(c,:);
Probability_HSI_EPLBP1=Probability_HSI_EPLBP(c,:);
Probability_LiDAR_EPLBP1=Probability_LiDAR_EPLBP(c,:);

% use the validation set to select the parameter a and b
map=zeros(100,100); 
a=[0.01:0.01:1];
b=[0.01:0.01:1];
OA_max=0;
for i=1:length(a)
    for j=1:length(b)
        tmp1=a(i)*b(j).*Probability_HSI1; 
        tmp2=b(j)*(1-a(i)).*Probability_HSI_EPLBP1;
        tmp3=(1-b(j)).*Probability_LiDAR_EPLBP1;
        tmp=tmp1+tmp2+tmp3;
        [value class] = max(tmp');        
        [OA, kA, CA, AA, errorMatrix] = calcError(testlabels1, class', classLabel);
        map(i,j)=OA;
        if OA>OA_max            
            OA_max=OA;
            A=[AA;OA_max;kA;CA];
            c=[a(i),b(j)];
        end
    end
end
figure;
mesh(a,b,map);
xlabel('a')
ylabel('b')
zlabel('OA')

%testing samples probability reconstruction using the parameter from validation set
tmp1=c(1)*c(2).*Probability_HSI; 
tmp2=c(2)*(1-c(1)).*Probability_HSI_EPLBP;
tmp3=(1-c(2)).*Probability_LiDAR_EPLBP;
tmp=tmp1+tmp2+tmp3;
[value class] = max(tmp');
[confusion, accuracy_CRT, TPR, AA1, KA1, FPR] = confusion_matrix_GCR(class, S);
OA_max=accuracy_CRT;
CA=TPR;
AA=AA1;
KA=KA1;
A=[AA;OA_max;KA;CA'];  

%% max voting
%If the results of the three classifiers are different, they are based on HSI, HSI_EPLBP, and LiDAR_EPLBP respectively.
voting=[maxPro_HSI;maxPro_HSI_EPLBP;maxPro_LiDAR_EPLBP]+1;
result=zeros(1,length(voting));
%HSI
for i=1:1:length(voting)
    if voting(2,i)==voting(3,i)& voting(1,i)~=voting(2,i)
    result(i)=voting(2,i);
    else
    result(i)=voting(1,i);
    end
end
%HSI_EPLBP
for i=1:1:length(voting)
    if voting(1,i)==voting(3,i)& voting(1,i)~=voting(2,i)
    result(i)=voting(1,i);
    else
    result(i)=voting(2,i);
    end
end
%LiDAR_EPLBP
for i=1:1:length(voting)
    if voting(1,i)==voting(2,i)&& voting(1,i)~=voting(3,i)
    result(i)=voting(2,i);
    else
    result(i)=voting(3,i);
    end
end

S=[];
for i=1:15
[N]=find(testlabels==i);
S=[S;length(N)];
end
[confusion, accuracy_CRT, TPR, AA1, KA1, FPR] = confusion_matrix_GCR(result, S);
OA_max=accuracy_CRT;
CA=TPR;
AA=AA1;
KA=KA1;
A=[AA;OA_max;KA;CA'];

%% Fusion method combination

% RNPRF.*RNDFF
a=0.61;b=0.72;
S=[];
classLabel=[1:15];
for i=1:15
[N]=find(testlabels==i);
S=[S;length(N)];
end
tmp1=a*b.*Probability_HSI; 
tmp2=b*(1-a).*Probability_HSI_EPLBP;
tmp3=(1-b).*Probability_LiDAR_EPLBP;
tmp=tmp1+tmp2+tmp3;
tmp1=tmp.*Probability_DF;
[value class] = max(tmp1');
[confusion, accuracy_CRT, TPR, AA1, KA1, FPR] = confusion_matrix_GCR(class, S);
OA_max=accuracy_CRT;
CA=TPR;
AA=AA1;
KA=KA1;
A=[AA;OA_max;KA;CA'];

% RNPMF.*RNDFF
S=[];
classLabel=[1:15];
for i=1:15
[N]=find(testlabels==i);
S=[S;length(N)];
end
tmp1=Probability_HSI.*Probability_HSI_EPLBP.*Probability_LiDAR_EPLBP.*Probability_DF;
[value class] = max(tmp1');
[confusion, accuracy_CRT, TPR, AA1, KA1, FPR] = confusion_matrix_GCR(class, S);
OA_max=accuracy_CRT;
CA=TPR;
AA=AA1;
KA=KA1;
A=[AA;OA_max;KA;CA'];

%RNPRF+RNDFF
a=0.61;b=0.72;
S=[];
classLabel=[1:15];
for i=1:15
[N]=find(testlabels==i);
S=[S;length(N)];
end
tmp1=a*b.*Probability_HSI; 
tmp2=b*(1-a).*Probability_HSI_EPLBP;
tmp3=(1-b).*Probability_LiDAR_EPLBP;
tmp=tmp1+tmp2+tmp3;
[value class] = max(tmp');
[confusion, accuracy_CRT, TPR, AA1, KA1, FPR] = confusion_matrix_GCR(class, S);
OA_max=accuracy_CRT;
CA=TPR;
AA=AA1;
KA=KA1;
A=[AA;OA_max;KA;CA'];

load('c.mat')% Use the location of the verification set samples to set the parameter ab.
testlabels1=testlabels(c);
Probability_DF1=Probability_DF(c,:);
tmp1=tmp(c,:);

map=zeros(1,100);
a=[0.01:0.01:1];
OA_max=0;
for i=1:length(a)  
        tmp1=a(i).*tmp1+(1-a(i)).*Probability_DF1; 
        [value class] = max(tmp1');
        [OA, kA, CA, AA, errorMatrix] = calcError(testlabels1, class', classLabel);        
        map(i)=accuracy_CRT;
        if accuracy_CRT>OA_max
            OA_max=accuracy_CRT;
            CA=TPR;
            AA=AA1;
            KA=KA1;
            c=a(i);
        end
end
tmp1=c.*tmp+(1-c).*Probability_DF; 
[value class] = max(tmp1');
[confusion, accuracy_CRT, TPR, AA1, KA1, FPR] = confusion_matrix_GCR(class, S);
OA_max=accuracy_CRT;
CA=TPR;
AA=AA1;
KA=KA1;
A=[AA;OA_max;KA;CA']; 
