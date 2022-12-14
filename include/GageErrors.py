CS_FALSE                         = 0  # Unnecessary operation
CS_SUCCESS                       = 1  # Sucessful Operation
CS_CONFIG_CHANGED                = 2  # Configuration Coerced
CS_ASYNC_SUCCESS                 = 3  # Asynchronous request succeeded
CS_SEGMENTINFO_EMPTY             = 4  # MinMax detection: Segment Info queue is empty
CS_STRUCTURE_PADDED              = 5  # Remote structure was larger than local structure and was padded with zeros
CS_STRUCTURE_TRUNCATED           = 6  # Remote structure was smaller than local structure and was truncated
CS_NOT_INITIALIZED               = -1  # Gage driver not initialized
CS_UNABLE_CREATE_RM              = -3  # Resource Manager unspecified failure
CS_INTERFACE_NOT_FOUND           = -4  # Resource Manager invalid interface
CS_HANDLE_IN_USE                 = -5  # Handle already locked
CS_INVALID_HANDLE                = -6  # Invalid handle
CS_INVALID_REQUEST               = -7  # Invalid request parameter
CS_NO_SYSTEMS_FOUND              = -8  # No digitizer system found
CS_MEMORY_ERROR                  = -9  # Failed memory allocation
CS_LOCK_SYSTEM_FAILED            = -10  # Lock system failed
CS_INVALID_STRUCT_SIZE           = -11  # Invalid structure size
CS_INVALID_STATE                 = -12  # Invalid action in current state
CS_INVALID_EVENT                 = -13  # Invalid event type
CS_INVALID_SHARED_REGION         = -14  # Cannot create shared memory region
CS_INVALID_FILENAME              = -15  # Invalid filename
CS_SHARED_MAP_UNAVAILABLE        = -16  # Cannot map shared memory region
CS_INVALID_START                 = -17  # Invalid start address
CS_INVALID_LENGTH                = -18  # Invalid buffer length
CS_SOCKET_NOT_FOUND              = -19  # Windows socket error
CS_SOCKET_ERROR                  = -20  # Resource Manager communication error
CS_NO_AVAILABLE_SYSTEM           = -21  # No digitizer system found with the requested requirements
CS_NULL_POINTER                  = -22  # Usage of NULL pointer
CS_INVALID_CHANNEL               = -23  # Invalid channel index
CS_INVALID_TRIGGER               = -24  # Invalid trigger index
CS_INVALID_EVENT_TYPE            = -25  # Invalid event type
CS_BUFFER_TOO_SMALL              = -26  # Buffer too small
CS_INVALID_PARAMETER             = -27  # Invalid parameter
CS_INVALID_SAMPLE_RATE           = -28  # Invalid sample rate
CS_NO_EXT_CLK                    = -29  # No external clock support
CS_SEG_COUNT_TOO_BIG             = -30  # Mulrec : Invalid size or count
CS_INVALID_SEGMENT_SIZE          = -31  # Invalid segment size
CS_DEPTH_SIZE_TOO_BIG            = -32  # Depth greater than segment size
CS_INVALID_CAL_MODE              = -33  # Invalid calibration mode
CS_INVALID_TRIG_COND             = -34  # Invalid trigger condition
CS_INVALID_TRIG_LEVEL            = -35  # Invalid trigger level
CS_INVALID_TRIG_SOURCE           = -36  # Invalid trigger source
CS_INVALID_EXT_TRIG              = -37  # No external trigger support
CS_INVALID_ACQ_MODE              = -38  # Invalid acquisition mode
CS_INVALID_IMPEDANCE             = -39  # Invalid impedance
CS_INVALID_GAIN                  = -40  # Invalid gain
CS_INVALID_COUPLING              = -41  # Invalid coupling
CS_BUFFER_NOT_ALIGNED            = -42  # Buffer not DWORD aligned
CS_PRETRIG_DEPTH_TOO_BIG         = -43  # Pretrigger depth exceeds the maximum supported value
CS_INVALID_TRIG_DEPTH            = -44  # Invalid trigger depth (not multiple of depth resolution)
CS_FUNCTION_NOT_SUPPORTED        = -45  # The feature, function or parameter is not supported by the current CompuScope model
CS_HARDWARE_TIMEOUT              = -46  # Timeout error
CS_INVALID_PARAMS_ID             = -47  # Invalid parameter ID
CS_INVALID_POINTER_BUFFER        = -48  # Invalid pointer
CS_CANNOT_LOCKDOWN_BUFFER        = -49  # DMA: Buffer too big
CS_DRIVER_ASYNC_REQUEST_BUSY     = -50  # Pending asynchronous operation
CS_INVALID_CHANNEL_COUNT         = -51  # Invalid channel count
CS_INVALID_TRIGGER_COUNT         = -52  # Invalid trigger count
CS_INVALID_SEGMENT               = -53  # Invalid segment
CS_INVALID_SEGMENT_COUNT         = -54  # Invalid segment count
CS_INVALID_CAPS_ID               = -55  # Invalid CapsID
CS_HANDLE_NOT_IN_USE             = -56  # Handle not in use
CS_INSUFFICIENT_RESOURCES        = -57  # Kernel memory allocation failed
CS_INVALID_TRANSFER_MODE         = -58  # Invalid transfer mode
CS_DRIVER_ASYNC_NOT_SUPPORTED    = -59  # Asynchronous calls not supported
CS_INVALID_TRIGGER_ENABLED       = -60  # Too many trigger engines
CS_NOT_TRIGGER_FROM_SAME_CARD    = -61  # Too many trigger masters
CS_INVALID_PRETRIGGER_DEPTH      = -62  # Pretrigger depth exceeds the maximum supported value
CS_INVALID_FW_VERSION            = -63  # Invalid firmware version
CS_INVALID_TRIGHOLDOFF           = -64  # Trigger Holdoff is invalid
CS_INVALID_TRIGDELAY             = -65  # Trigger Delay is invalid
CS_INVALID_STREAMING_BUFFER      = -66  # Invalid Streaming buffer
CS_HW_FIFO_OVERFLOW              = -67  # Error HW FIFO overlow
CS_INVALID_CARD                  = -68  # Invalid card index
CS_INVALID_TOKEN                 = -69  # Invalid token for Asynchronous transfer
CS_MASTERSLAVE_DISCREPANCY       = -70  # Discrepancies between Master and Slave boards
CS_INVALID_FIR_MULTIPLIER        = -71  # The mutiplier parameter of FIR structure is invalid
CS_MINMAXDETECTQUEUE_INVALID     = -72  # The MinMax driver internal queue is invalid
CS_ALLTRIGGERENGINES_USED        = -73  # Cannot find a free trigger engine dedicated for channel or card
CS_MULREC_RAWDATA_TOOBIG         = -74  # Mulrec raw data exeeds 4G bytes
CS_INVALID_FILTER                = -75  # Invalid filter value
CS_INVALID_POSITION              = -76  # Invalid DC offset
CS_EXT_CLK_OUT_OF_RANGE          = -77  # External clock frequency is not supported by this model
CS_EXT_CLK_NOT_PRESENT           = -78  # External clock frequency error or external clock signal is not present.
CS_RE_INIT_FAILED                = -79  # Cannot reinitialize when systems are active
CS_CHANNEL_PROTECT_FAULT         = -80  # Incomplete acquisition due to user Abort request or Channel Proctection Fault.
CS_INVALID_NUM_OF_AVERAGE        = -81  # Invalid number of averages
CS_INVALID_TIMESTAMP_CLOCK       = -82  # Invalid time stamp clock
CS_ASYNCTRANSFER_ABORTED         = -83  # Asynchronous data transfer has been aborted
CS_STREAM_ERROR_CIRCULAR_BUFFER  = -84  # Circular buffer for streaming is not allocated
CS_STREAM_ERROR_CREATEFILE       = -85  # Error create file for streaming
CS_STREAM_ERROR_WRITEFILE        = -86  # Error write to stream file
CS_TRANSFER_DATA_TIMEOUT         = -87  # Timeout on data transfer
CS_DEVIOCTL_ERROR                = -88  # DeviceIoControl request error
CS_OVERLAPPED_ERROR              = -89  # No more OVERLAPPED structure
CS_CREATEEVENT_ERROR             = -90  # CreateEvent error
CS_FLASHSTATE_ERROR              = -91  # Error on Flash state
CS_FLASH_SECTORCROSS_ERROR       = -92  # Error on Flash cross boundary
CS_FLASH_SECTORERASE_ERROR       = -93  # Error on Flash Erase Sector
CS_FLASH_INVALID_SECTOR          = -94  # Flash operation is done on an invalid sector
CS_FLASH_DATAREAD_ERROR          = -95  # Flash data read error
CS_FLASH_DATAWRITE_ERROR         = -96  # Flash data write error
CS_SYSTEM_NOT_INITIALIZED        = -97  # Acquisition system is  not initialized
CS_POWERSTATE_ERROR              = -98  # Power State error. Please close then restart the application.
CS_SEGMENTSIZE_TOO_BIG           = -99  # Segment size exceeds the maximum supported value
CS_INVALID_FRM_CMD               = -100  # Unrecognized firmware command
CS_FRM_NO_RESPONSE               = -101  # Nios timeout
CS_INVALID_DAC_ADDR              = -102  # Invalid DAC address
CS_INVALID_EDGE                  = -103  # Invalid delay line edge
CS_INVALID_SELF_TEST             = -104  # Invalid self test mode
CS_NIOS_FAILED                   = -105  # Nios failed on reset
CS_NO_INTERRUPT                  = -106  # No system support for interrupts
CS_ADDONINIT_ERROR               = -107  # Addon initialization failure
CS_INVALID_ADC_ADDR              = -108  # Invalid ADC address 
CS_ADC_ACCESS_ERROR              = -109  # Access to ADC failed
CS_FLASH_TIMEOUT                 = -110  # Access to Flash or Eeprom timeout
CS_FLASH_BUFFER_BOUNDARY_ERROR   = -111  # Error on the buffer address when reading/writting flash
CS_FLASH_BUFFER_SIZE_ERROR       = -112  # Error on the buffer size when reading/writting flash
CS_FLASH_ADDRESS_ERROR           = -113  # Error on the address of the flash
CS_FLASH_ERASESECTOR_ERROR       = -114  # Error on erasing the flash sector
CS_INVALID_CARD_COUNT            = -120  # Invalid board count
CS_DAC_CALIB_FAILURE             = -200  # Calibration DAC failure on voltage
CS_CAL_BUSY_TIMEOUT              = -201  # Busy timeout in calibration sequence
CS_TIMING_CAL_FAILED             = -202  # Timing calibration failure
CS_COARSE_OFFSET_CAL_FAILED      = -203  # Coarse offset calibration failed on channel %d
CS_FINE_OFFSET_CAL_FAILED        = -204  # Fine offset calibration failed on channel %d
CS_GAIN_CAL_FAILED               = -205  # Gain calibration failed on channel %d
CS_POSITION_CAL_FAILED           = -206  # Position calibration failed on channel %d
CS_CALIB_DAC_OUT_OF_RANGE        = -207  # Calibration DAC is out of range
CS_CALIB_ADC_CAPTURE_FAILURE     = -208  # Failure to capture on calibration ADC
CS_CALIB_MEAS_ADC_FAILURE        = -209  # Failure to read reference ADC
CS_CALIB_ADC_READ_FAILURE        = -210  # Failure to read calibration ADC
CS_CHANNELS_NOTCALIBRATED        = -211  # Some channels are not calibrated
CS_MASTERSLAVE_CALIB_FAILURE     = -212  # Master/Slave calibration failed
CS_EXTTRIG_CALIB_FAILURE         = -213  # Ext Trigger calibration failed
CS_INVALID_DACCALIBTABLE         = -214  # DAC CalibTable in Eeprom is invalid
CS_ADCALIGN_CALIB_FAILURE        = -215  # ADCs alignment calibration failed
CS_CALIB_REF_FAILURE             = -216  # Calibration source reference failed
CS_FAST_CALIB_FAILURE            = -217  # Fast calibration failed
CS_ADC_DPA_FAILURE               = -218  # DPA of the ADC failed
CS_EXT_TRIG_DPA_FAILURE          = -219  # Alignment of External trigger failed
CS_CLK_NOT_LOCKED                = -220  # Main PLL lost clock lock
CS_INVALID_CALIB_BUFFER          = -221  # Buffer for calibration is not allocated.
CS_NULL_OFFSET_CAL_FAILED        = -222  # Null offset calibration failed on channel %d
CS_ADC_SKEW_CAL_FAILED           = -223  # Intra core skew calibration failed on channel %d
CS_ADC_PHASE_CAL_FAILED          = -224  # Intra ADC skew calibration failed on channel %d
CS_ACCAL_NOT_LOCKED              = -225  # AC Calibration PLL lost clock lock
CS_LVDS_ALIGNMENT_FAILED         = -226  # LVDS Alignement failed
CS_OFFSET_ADJUST_FAILED          = -227  # DC Offset adjustment failed on channel %d
CS_ADC_IQ_GAIN_CALIB_FAILED      = -228  # ADC I/Q Gain calibration failed
CS_ADC_IQ_OFFSET_CALIB_FAILED    = -229  # ADC I/Q Offset calibration failed
CS_DC_LEVEL_FREEZE_FAILED        = -230  # DC Level Freeze failed
CS_GAIN_CROSS_CHECK_FAILED       = -231  # Channel gain cross check boundary failed. (Exceed 1 db)
CS_NVRAM_NOT_INIT                = -300  # nvRam content is bad
CS_FLASH_NOT_INIT                = -301  # FLASH content is bad
CS_EEPROM_NOT_INIT               = -302  # Eeprom content is bad
CS_ADDON_NOT_CONNECTED           = -303  # Addon board is not connected
CS_MS_BRIDGE_FAILED              = -304  # Master/slave connector test has failed
CS_GIO_DETECT_FAILED             = -305  # Failed to determine the GIO access mode
CS_ADDON_FPGA_LOAD_FAILED        = -306  # Failed to load addon FPGA
CS_EEPROM_WRITE_TIMEOUT          = -307  # Timeout when writting to eeprom
CS_CSI_FILE_ERROR                = -407  # CompuScope information file is missing or corrupted
CS_INVALID_DIRECTORY             = -408  # Directory is invalid
CS_DISK_STREAM_NOT_INITIALIZED   = -409  # CompuScope DiskStream subsystem has not been initialized
CS_INVALID_DISK_STREAM_ACQ_COUNT = -410  # CompuScope DiskStream acquisition count is invalid
CS_INVALID_SIG_HEADER            = -411  # Invalid SIG file header
CS_FWUPDATED_SHUTDOWN_REQUIRED   = -412  # The PC needs to be shutdown then reboot after updated FW.
CS_INVALID_DATAPACKING           = -413  # Invalid Data Packing mode
CS_STM_TRANSFER_ABORTED          = -800  # Stream transfer has been aborted
CS_STM_FIFO_OVERFLOW             = -801  # Stream Fifo overflow. 
CS_STM_INVALID_TRANSFER_SIZE     = -802  # Error on stream transfer size. 
CS_STM_COMPLETED                 = -803  # Stream acquisition completed.
CS_STM_TRANSFER_TIMEOUT          = -804  # Stream transfer timeout
CS_HISTOGRAM_FULL                = -805  # Histogram full
CS_STM_INVALID_BUFFER            = -806  # The buffer for streaming is invalid.
CS_DDC_CORE_CONFIG_ERROR         = -807  # Error on DDC core configuration.
CS_STM_TOTALDATA_SIZE_INVALID    = -808  # Total data size of streaming acquisition is invalid
CS_OCT_CORE_CONFIG_ERROR         = -809  # Error on OCT core configuration.
CS_OCT_INVALID_CONFIG_PARAMS     = -810  # OCT Config Params is invalid.
CS_REMOTE_SOCKET_ERROR           = -850  # Remote instrument communication error
CS_MISC_ERROR                    = -32767  # Miscellaneous Error
