-- This file is part of https://github.com/SensorsINI/dnd_hls.
-- This intellectual property is licensed under the terms of the project license available at the root of the project.
--------------------------------------------------------------------------------
-- Company: ini
-- Engineer: Alejandro Linares
--
-- Create Date:    14:16:08 06/17/14
-- Design Name:    
-- Module Name:    okt_wsaer2caviar 
-- Project Name:   VISUALIZE
-- Target Device:  Latticed LFE3-17EA-7ftn256i
-- Tool versions:  Diamond x64 3.0.0.97
-- Description: Module to convert word-serial AER from DAViS to CAVIAR parallel AER
--
-- 
--------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.numeric_std.all;

---- Uncomment the following library declaration if instantiating
---- any Xilinx primitives in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity okt_wsaer2caviar is
	generic(
		WSAER_BIT_WIDTH     : integer := 10;
		CAVIAR_BIT_WIDTH    : integer := 17; -- This value must be (WSAER_BIT_WIDTH-2)*2 at least
		ROW_DELAY_BIT_WIDTH : integer := 5
	);
	port(
		wsaer_data   : in  std_logic_vector(WSAER_BIT_WIDTH - 1 downto 0); --bit 0 is polarity and bit 9 row / column.
		wsaer_req_n  : in  std_logic;   --active low and synchronized
		wsaer_ack_n  : out std_logic;   --Supossed to be active low

		-- clock and reset inputs
		clk          : in  std_logic;
		rst_n        : in  std_logic;
		row_delay    : in  unsigned(ROW_DELAY_BIT_WIDTH - 1 downto 0);
		-- AER monitor interface
		caviar_ack_n : in  std_logic;   -- needs synchronization
		cavir_req_n  : out std_logic;
		caviar_data  : out std_logic_vector(CAVIAR_BIT_WIDTH - 1 downto 0));

end okt_wsaer2caviar;

architecture Structural of okt_wsaer2caviar is
	--signal dwsaer_req_n,d1wsaer_req_n,d2wsaer_req_n, wsaer_data9, last9: std_logic;
	--signal latched_ev : std_logic_vector (9 downto 0);
	type state is (idle, req, row, row_ack, col, col_ack, send, send_ack);
	signal cs, ns : state;
	signal cnt    : unsigned(ROW_DELAY_BIT_WIDTH - 1 downto 0);

	-- DEBUG
	attribute MARK_DEBUG : string;
	attribute MARK_DEBUG of cs, ns, cnt: signal is "TRUE";
begin
	process(clk, rst_n)
	begin
		if (rst_n = '0') then
			caviar_data <= (others => '0');
			cs          <= idle;
			cnt         <= (others => '0');

		elsif (clk'event and clk = '1') then
			cs <= ns;
			case cs is
				when idle => 
					cnt <= (others => '0');

				when row =>
					cnt <= cnt + 1;
					if (cnt = row_delay) then
						caviar_data(CAVIAR_BIT_WIDTH-1 downto WSAER_BIT_WIDTH-1) <= wsaer_data(WSAER_BIT_WIDTH - 3 downto 0); -- @suppress "Incorrect array size in assignment: expected (<CAVIAR_BIT_WIDTH + -1*WSAER_BIT_WIDTH + 1>) but was (<WSAER_BIT_WIDTH - 2>)"
					end if;

				when col =>
					caviar_data(WSAER_BIT_WIDTH - 2 downto 0) <= wsaer_data(WSAER_BIT_WIDTH - 2 downto 0);
					
				when others => null;
			end case;
		end if;
	end process;

	process(cs, wsaer_req_n, caviar_ack_n, cnt, row_delay, wsaer_data)
	begin
		cavir_req_n <= '1';
		wsaer_ack_n <= '1';
		case cs is
			when idle =>
				if wsaer_req_n = '0' then
					ns <= req;
				else
					ns <= idle;
				end if;

			when req =>
				if wsaer_data(WSAER_BIT_WIDTH - 1) = '0' then
					ns <= row;
				else
					ns <= col;
				end if;

			when row =>
				wsaer_ack_n <= '1';
				if (cnt = row_delay) then
					ns <= row_ack;
				else
					ns <= row;
				end if;

			when row_ack =>
				wsaer_ack_n <= '0';
				if wsaer_req_n = '0' then
					ns <= row_ack;
				else
					ns <= idle;
				end if;

			when col =>
				wsaer_ack_n <= '1';
				ns          <= col_ack;

			when col_ack =>
				wsaer_ack_n <= '0';
				if wsaer_req_n = '0' then
					ns <= col_ack;
				else
					ns <= send;
				end if;

			when send =>
				cavir_req_n <= '0';
				if caviar_ack_n = '0' then
					ns <= send_ack;
				else
					ns <= send;
				end if;

			when send_ack =>
				if caviar_ack_n = '1' then
					ns <= idle;
				else
					ns <= send_ack;
				end if;
				--when others => ns <= idle;
		end case;
	end process;
end Structural;

