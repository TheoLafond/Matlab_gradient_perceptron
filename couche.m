classdef  couche < handle
   properties
      w;
      b;
      btemp;
      wtemp;
   end
   methods
       function init(obj,ne,nc)
         obj.w = 2*rand(ne,nc)-1;
         obj.b = zeros(1,nc);
         
      end
      function p = phi(obj,z)
          p=exp(z)./(1+exp(z));
      end
      function z = signal_activation(obj,x)
         z = x*obj.w+obj.b;
      end
      function p = y(obj,x)
         p = obj.phi(obj.signal_activation(x));
      end
      function grade=grade(obj,c,data,Y1,Y2,w2)
          N=size(data,1);
          grade=1/(2*N)*transpose(data)*((Y1-Y1.^2).*(((Y2-c).*(Y2-Y2.^2))*transpose(w2)));
      end
      function grads=grads(obj,c,data,Y)
          N=size(data,1);
          grads=1/(2*N)*transpose(data)*((Y-c).*(Y-Y.^2));
      end
      function grabe=gradbe(obj,c,data,Y1,Y2,w2)
          N=size(data,1);
          grabe=1/(2*N)*sum((Y1-Y1.^2).*(((Y2-c).*(Y2-Y2.^2))*transpose(w2)));
      end
      function grabs=gradbs(obj,c,Y)
          N=size(Y,1);
          grabs=1/(2*N)*sum((Y-c).*(Y-Y.^2));
      end
   end
end
