classdef mapCrop_STrand_Ap < nnet.layer.Layer

    properties
        segmentLen
    end

    
    methods
        function layer = mapCrop_STrand_Ap(segmentLen, name)
            layer.Name = name;
            layer.Description = "Map crop layer";
            layer.segmentLen = segmentLen;
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
                        
            rand_range = size(X, 2) - layer.segmentLen;
            rand_dot = ceil(rand_range*rand);
            
            % Randomly select a part of data
            Z = FeatureExtract_STrans_Ap_4D_GPU(X(:,rand_dot: rand_dot + layer.segmentLen - 1,:,:));    

        end
        
        function [dLdX] = backward(layer, X, ~, dLdZ, ~)
            % [dLdX, dLdAlpha] = backward(layer, X, ~, dLdZ, ~)
            % backward propagates the derivative of the loss function
            % through the layer.
            %
            % Inputs:
            %         layer    - Layer to backward propagate through 
            %         X        - Input data 
            %         dLdZ     - Gradient propagated from the deeper layer 
            % Outputs:
            %         dLdX     - Derivative of the loss with respect to the
            %                    input data
            
            dLdX = X;
        end
    end
end
