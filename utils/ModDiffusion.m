classdef ModDiffusion

    methods (Static)
        
        % --- Forward Modulo Diffusion ---
        function Cipher = diffusion_fwd_mod(Plain, Key, IV)
            len = length(Plain);
            Cipher = zeros(len, 1, 'uint8');
            
            P_double = double(Plain);
            K_double = double(Key);
            prev = double(IV); % Use dynamic IV
            
            for i = 1:len
                % C(i) = (P(i) + C(i-1) + K(i)) mod 256
                temp = P_double(i) + prev + K_double(i);
                val = mod(temp, 256);
                Cipher(i) = uint8(val);
                prev = val;
            end
        end

        % --- Backward Modulo Diffusion (Tail to Head) ---
        function Cipher = diffusion_bwd_mod(Plain, Key, IV)
            len = length(Plain);
            Cipher = zeros(len, 1, 'uint8');
            
            P_double = double(Plain);
            K_double = double(Key);
            next_val = double(IV); % Use dynamic IV
            
            for i = len:-1:1
                % C(i) = (P(i) + C(i+1) + K(i)) mod 256
                temp = P_double(i) + next_val + K_double(i);
                val = mod(temp, 256);
                Cipher(i) = uint8(val);
                next_val = val;
            end
        end

        % --- Inverse Backward Modulo Diffusion ---
        function Plain = inv_diffusion_bwd_mod(Cipher, Key, IV)
            len = length(Cipher);
            Plain = zeros(len, 1, 'uint8');
            
            C_double = double(Cipher);
            K_double = double(Key);
            next_val = double(IV); % Must match encryption IV
            
            for i = len:-1:1
                curr_c = C_double(i);
                % P(i) = (C(i) - C(i+1) - K(i)) mod 256
                temp = curr_c - next_val - K_double(i);
                val = mod(temp, 256);
                Plain(i) = uint8(val);
                next_val = curr_c; % Feedback ciphertext
            end
        end

        % --- Inverse Forward Modulo Diffusion ---
        function Plain = inv_diffusion_fwd_mod(Cipher, Key, IV)
            len = length(Cipher);
            Plain = zeros(len, 1, 'uint8');
            
            C_double = double(Cipher);
            K_double = double(Key);
            prev = double(IV); % Must match encryption IV
            
            for i = 1:len
                curr_c = C_double(i);
                % P(i) = (C(i) - C(i-1) - K(i)) mod 256
                temp = curr_c - prev - K_double(i);
                val = mod(temp, 256);
                Plain(i) = uint8(val);
                prev = curr_c; % Feedback ciphertext
            end
        end
        
    end
end