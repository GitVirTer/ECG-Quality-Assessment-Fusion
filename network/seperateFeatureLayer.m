classdef seperateFeatureLayer < nnet.layer.Layer

    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end

    methods
        function layer = seperateFeatureLayer(name)
            layer.Name = name;
            
            layer.Description = "Seperate Layer";
            layer.NumOutputs = 2;
            layer.OutputNames = {'HE', 'DL'};
        end
        
        function varargout = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            numHE = 3;
            varargout{1} = X(:,1:numHE,:,:);
            varargout{2} = X(:,numHE+1:end,:,:);

        end
        
        function [dLdX] = backward(layer, varargin)
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
            
            dLdX = zeros(size(varargin{1}), 'like', varargin{1});

        end
    end
end

