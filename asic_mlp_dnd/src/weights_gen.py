import numpy as np
np.random.seed(0)

N1 = 98
N2 = 10
W_K = 4
W_OUT = 16

w1_mag = np.random.randint(-2**W_K/2, 2**W_K/2-1, (N2,N1//2+1))
w1_pol = np.random.randint(-2**W_K/2, 2**W_K/2-1, (N2,N1//2+1))
# np.savetxt("weights_n1.mem", w1.flatten(), fmt="%x")

w2 = np.random.randint(0, 2**W_K, N2+1)
# np.savetxt("weights_n2.mem", w1, fmt="%x")

tan = np.random.randint(0, 2**W_OUT, 2**W_K)

'''
WRITE WEIGHTS
'''

w1_mag_all = ""
for row in w1_mag:
  row_text = ""
  for e in row:
    s = '-' if e < 0 else ' '
    row_text += f"{s}{W_K}'d{abs(e)}".ljust(10) + ", "
  row_text = f"'{{ {row_text[:-2]} }}"
  w1_mag_all += f'''    {row_text},\n'''

w1_pol_all = ""
for row in w1_pol:
  row_text = ""
  for e in row:
    s = '-' if e < 0 else ' '
    row_text += f"{s}{W_K}'d{abs(e)}".ljust(10) + ", "
  row_text = f"'{{ {row_text[:-2]} }}"
  w1_pol_all += f'''    {row_text},\n'''

n2_row_text = ""
for e in w2:
  s = '-' if e < 0 else ' '
  n2_row_text += f"\n    {s}{W_K}'d{abs(e)}".ljust(10) + ","

tan_row_text = ""
for e in tan:
  s = '-' if e < 0 else ' '
  tan_row_text += f"\n    {s}{W_OUT}'d{abs(e)}".ljust(10) + ","

with open("luts.sv", "w") as f:
  f.write(f'''
`timescale 1ns/1ps

module luts #(N1={N1}, N2={N2}, W_K={W_K}, W_OUT={W_OUT})(
  output wire [N2-1:0][N1/2:0][W_K-1:0] weights_n1_mag,
  output wire [N2-1:0][N1/2:0][W_K-1:0] weights_n1_pol,
  output wire [N2  :0][W_K-1:0] weights_n2,
  output wire [2**W_K-1:0][W_OUT-1:0] tanh 
);
  assign weights_n1_mag = '{{
{w1_mag_all[:-2]}
  }};

  assign weights_n1_pol = '{{
{w1_pol_all[:-2]}
  }};

  assign weights_n2 = '{{{n2_row_text[:-1]}
  }};

  assign tanh = '{{{tan_row_text[:-1]}
  }};
endmodule
''')