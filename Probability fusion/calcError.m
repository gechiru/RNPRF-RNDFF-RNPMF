function [OA, kA, CA, AA, errorMatrix] = calcError(Y, Ypre, classLabel)
% CA: class_accuracy
% AA: average_accuracy
% OA: overall_accuracy
% kA: kappa_accuracy
% Y : true labels
% Ypre: predicted labels

nClass = length(classLabel);
nrPixelsPerClass = zeros(nClass, 1); % number of samples in each class
errorMatrix = zeros(nClass, nClass);
for i = 1 : nClass
    indi = find (Y == classLabel(i));
    nrPixelsPerClass(i) = length(indi);
    for j = 1 : nClass
        indj = find (Ypre == classLabel(j));
        errorMatrix(i,j) = length( intersect(indi, indj) );
    end
end
diagVector = diag(errorMatrix);
CA = diagVector ./ (nrPixelsPerClass+eps);  % class_accuracy
AA = mean(CA);                              % average_accuracy
OA = sum(Ypre==Y) / length(Y);              % overall_accuracy
kA = (sum(errorMatrix(:))*sum(diag(errorMatrix)) - sum(errorMatrix)*sum(errorMatrix,2))...
    /(sum(errorMatrix(:))^2 -  sum(errorMatrix)*sum(errorMatrix,2));


