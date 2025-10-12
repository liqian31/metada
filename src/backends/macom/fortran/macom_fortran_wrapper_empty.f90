module macom_fortran_wrapper
  use iso_c_binding
  implicit none

  private ! Default to private, only expose BIND(C) interfaces

contains

  !-----------------------------------------------------------------------------
  ! Empty implementations for when MACOM_MODE_ENABLED is FALSE
  !-----------------------------------------------------------------------------
  subroutine c_macom_initialize_mpi(comm_cpp) &
    bind(C, name="c_macom_initialize_mpi")
    integer(C_INT), value, intent(in) :: comm_cpp
    ! Empty - MACOM mode disabled
  end subroutine c_macom_initialize_mpi

  subroutine c_macom_get_mpi_rank(rank) &
    bind(C, name="c_macom_get_mpi_rank")
    integer(C_INT), intent(out) :: rank
    rank = 0
  end subroutine c_macom_get_mpi_rank

  subroutine c_macom_get_mpi_size(size) &
    bind(C, name="c_macom_get_mpi_size")
    integer(C_INT), intent(out) :: size
    size = 1
  end subroutine c_macom_get_mpi_size

  subroutine c_macom_finalize_mpi() &
    bind(C, name="c_macom_finalize_mpi")
    ! Empty - MACOM mode disabled
  end subroutine c_macom_finalize_mpi

  subroutine c_macom_read_namelist() &
    bind(C, name="c_macom_read_namelist")
    ! Empty - MACOM mode disabled
  end subroutine c_macom_read_namelist

  subroutine c_get_macom_config_flags(mitice, restart, assim, &
                                      init_iter, max_iter) &
    bind(C, name="c_get_macom_config_flags")
    logical(C_BOOL), intent(out) :: mitice, restart, assim
    integer(C_INT), intent(out) :: init_iter, max_iter
    ! Return default values when MACOM mode is disabled
    mitice = .false.
    restart = .false.
    assim = .false.
    init_iter = 0
    max_iter = 1
  end subroutine c_get_macom_config_flags

  subroutine c_macom_mpi_send_info_comp_to_io() &
    bind(C, name="c_macom_mpi_send_info_comp_to_io")
    ! Empty - MACOM mode disabled
  end subroutine c_macom_mpi_send_info_comp_to_io

  subroutine c_macom_misc_run_info_open() &
    bind(C, name="c_macom_misc_run_info_open")
    ! Empty - MACOM mode disabled
  end subroutine c_macom_misc_run_info_open

  subroutine c_macom_init_csp() &
    bind(C, name="c_macom_init_csp")
    ! Empty - MACOM mode disabled
  end subroutine c_macom_init_csp

  subroutine c_macom_restart_and_assim() &
    bind(C, name="c_macom_restart_and_assim")
    ! Empty - MACOM mode disabled
  end subroutine c_macom_restart_and_assim

  subroutine c_macom_run_csp_step() &
    bind(C, name="c_macom_run_csp_step")
    ! Empty - MACOM mode disabled
  end subroutine c_macom_run_csp_step

  subroutine c_macom_csp_io_main() &
    bind(C, name="c_macom_csp_io_main")
    ! Empty - MACOM mode disabled
  end subroutine c_macom_csp_io_main

  subroutine c_macom_initialize_mitice() &
    bind(C, name="c_macom_initialize_mitice")
    ! Empty - MACOM mode disabled
  end subroutine c_macom_initialize_mitice

  subroutine c_macom_mitice_init_all() &
    bind(C, name="c_macom_mitice_init_all")
    ! Empty - MACOM mode disabled
  end subroutine c_macom_mitice_init_all

  subroutine c_macom_finalize_mitice() &
    bind(C, name="c_macom_finalize_mitice")
    ! Empty - MACOM mode disabled
  end subroutine c_macom_finalize_mitice

  subroutine c_macom_4var_da_main() &
    bind(C, name="c_macom_4var_da_main")
    ! Empty - MACOM mode disabled
  end subroutine c_macom_4var_da_main

end module macom_fortran_wrapper
