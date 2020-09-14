import time
import serial
import serial.tools.list_ports as slp


class DigitalIO(object):
    def __init__(self, name):
        self.o_state = 0
        self.i_state = 0


    def _apply_mask(self, value, mask):
        self.o_state = (self.o_state & ~mask) | (value & mask)


    def read(self):
        raise NotImplementedError()
        

    def write(self, value, mask):
        raise NotImplementedError()


    def set_bit(self, mask):
        return self.write(mask, mask)
        

    def reset_bit(self, mask):
        return self.write(0, mask)



class PhysicalDigitalIO(DigitalIO):
    pass



class ParallelPortIO(PhysicalDigitalIO):
    @classmethod
    def scan(cls):
        return []


class SerialPortIO(PhysicalDigitalIO):
    @classmethod
    def scan(cls):
        ports = serial.tools.list_ports.comports()
        ports = [port[0] if isinstance(port, tuple) else port.device for port in ports]
        return ports

    def __init__(self, name):
        super(SerialPortIO, self).__init__(name)
        self.device = serial.Serial(port=name, xonxoff=True, rtscts=False)
        

    def write(self, value, mask):
        if value == 0:
            self.device.setRTS(0)
        else:
            self.device.setRTS(1)



class FTDI245IO(PhysicalDigitalIO):
    @classmethod
    def scan(cls):
        return []


    def __init__(self, name):
        super(FTDI245IO, self).__init__(name)



class DigitalIOLoggingEmulator(DigitalIO):
    @classmethod
    def scan(cls):
        return ["Logging emulator"]


    def __init__(self, name):
        super(DigitalIOLoggingEmulator, self).__init__(name)


    def read(self):
        return 0
        

    def write(self, value, mask=0xFFFFFFFF):
        self._apply_mask(value, mask)
        print("DigitalIO: OUT {}".format(self.o_state))
        return self.o_state



class DigitalIOFactory(object):
    io_interfaces = None  # list of (name, class)

    @classmethod
    def scan(cls):
        cls.io_interfaces = \
            [(name, DigitalIOLoggingEmulator) for name in DigitalIOLoggingEmulator.scan()] + \
            [(name, ParallelPortIO) for name in ParallelPortIO.scan()] + \
            [(name, FTDI245IO) for name in FTDI245IO.scan()] + \
            [(name, SerialPortIO) for name in SerialPortIO.scan()]
        return cls.io_interfaces


    @classmethod
    def enumerate(cls, baseclass=DigitalIO):
        return [i[0] for i in cls.io_interfaces if  issubclass(i[1], baseclass)]


    @classmethod
    def create(cls, name):
        for i in cls.io_interfaces:
            if i[0] == name:
                return i[1](name)
        raise ValueError("Can not find digital interface '{}'".format(name))


DigitalIOFactory.scan()
    


if __name__ == "__main__":
    # Emulator
    names = DigitalIOFactory.enumerate()
    print("Available digital IO ports: {}".format(names))
    io = DigitalIOFactory.create(DigitalIOFactory.enumerate(DigitalIOLoggingEmulator)[0])
    assert io.write(0x03, 0x09) == 1
    assert io.set_bit(0x02) == 3
    assert io.reset_bit(0x01) == 2

    # COM port
    devices = DigitalIOFactory.enumerate(SerialPortIO)
    print("Available serial ports: {}".format(devices))
    if len(devices) > 0:
        device = devices[0]
        print("Using {}".format(device))
        io = DigitalIOFactory.create(device)
        for i in range(10):
            io.reset_bit(0x01)
            time.sleep(0.05)
            io.set_bit(0x01)
            time.sleep(0.05)        
    else:
        print("No serial ports found!")
    print("END")
    