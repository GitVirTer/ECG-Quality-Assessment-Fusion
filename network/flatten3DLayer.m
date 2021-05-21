classdef flatten3DLayer < nnet.layer.Layer
    
    methods
        function layer = flatten3DLayer(name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "DO NOTHING";
            % Layer constructor function goes here.
        end
        
        function [Z, memory] = forward(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
                
            sizeX = size(X);
            sizeZ = sizeX;
            sizeZ(3) = sizeX(1)*sizeX(2)*sizeX(3);
            sizeZ([1,2]) = 1;
            Z = reshape(X, sizeZ);
%             Z = squeeze(X);
            memory = sizeX;
            
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.

            sizeX = size(X);
            sizeZ = sizeX;
            sizeZ(3) = sizeX(1)*sizeX(2)*sizeX(3);
            sizeZ([1,2]) = 1;
            Z = reshape(X, sizeZ);
        end
        
        function [dLdX] = backward(layer, ~, ~, dLdZ, memory)
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

            sizeX = memory;
            dLdX = reshape(dLdZ, sizeX);

        end

    end
end
