
import os
from dynamixel_sdk import *

import sys
import platform
import numpy as np
from pathlib import Path

def run():

    # Control table address
    ADDR_MX_TORQUE_ENABLE      = 24               # Control table address is different in Dynamixel model
    ADDR_MX_GOAL_POSITION = 30
    ADDR_GOAL_VELOCITY    = 32
    ADDR_MX_PRESENT_POSITION   = 36
    ADDR_MX_PRESENT_SPEED      = 38

    # Protocol version
    PROTOCOL_VERSION            = 1.0               # See which protocol version is used in the Dynamixel

    # Default setting
    BAUDRATE                    = 1000000             # Dynamixel default baudrate : 57600
    DEVICENAME                  = '/dev/ttyUSB0'    # Check which port is being used on your controller
                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"
    TORQUE_ENABLE               = 1                 # Value for enabling the torque
    TORQUE_DISABLE              = 0                 # Value for disabling the torque
    DXL_MINIMUM_POSITION_VALUE  = 0                 # Dynamixel will rotate between this value
    DXL_MAXIMUM_POSITION_VALUE  = 1023              # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
    DXL_MOVING_STATUS_THRESHOLD = 20                # Dynamixel moving status threshold

    # Initialize PortHandler instance
    # Set the port path
    # Get methods and members of PortHandlerLinux or PortHandlerWindows
    portHandler = PortHandler(DEVICENAME)

    # Initialize PacketHandler instance
    # Set the protocol version
    # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    # Open port
    if portHandler.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        print("Press any key to terminate...")
        getch()
        quit()

    # Set port baudrate
    if portHandler.setBaudRate(BAUDRATE):
        print("Succeeded to change the baudrate")
    else:
        print("Failed to change the baudrate")
        print("Press any key to terminate...")
        getch()
        quit()

    # Enable Dynamixel Torque
    dxl_comm_result1, dxl_error1 = packetHandler.write1ByteTxRx(portHandler, 5, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    dxl_comm_result2, dxl_error2 = packetHandler.write1ByteTxRx(portHandler, 7, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    print(f"5 : {dxl_comm_result1}, 7 : {dxl_comm_result2}")

    GoalAngles = np.array([-90,0], float)

    while True:
        dxl_comm_result1, dxl_error1 = packetHandler.write2ByteTxRx(portHandler, 5, ADDR_MX_GOAL_POSITION, int(512*(GoalAngles[0]/150 +1)))
        dxl_comm_result2, dxl_error2 = packetHandler.write2ByteTxRx(portHandler, 7, ADDR_MX_GOAL_POSITION, int(512*(GoalAngles[1]/150 +1)))
        print(f"5 : {dxl_comm_result1}, 7 : {dxl_comm_result2}")

if __name__ == "__main__":
    run()
