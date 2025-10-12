module macom_fortran_wrapper
  use iso_c_binding
  use macom_logger
  ! 4D-Var DA mode specific imports
  use mod_nmefc_macom
  use mod_a4d_liwei
  ! Do NOT import mod_csp_basic here to avoid conflicts in 4D-Var DA mode
  
  implicit none

  private ! Default to private, only expose BIND(C) interfaces

contains

  !-----------------------------------------------------------------------------
  ! I. MPI Management (minimal implementation for 4D-Var DA mode)
  !-----------------------------------------------------------------------------
  subroutine c_macom_initialize_mpi(comm_cpp) &
    bind(C, name="c_macom_initialize_mpi")
    integer(C_INT), value, intent(in) :: comm_cpp
    ! Empty - MPI initialization handled by 4D-Var DA main
  end subroutine c_macom_initialize_mpi

  subroutine c_macom_get_mpi_rank(rank) &
    bind(C, name="c_macom_get_mpi_rank")
    integer(C_INT), intent(out) :: rank
    rank = mpi_rank
  end subroutine c_macom_get_mpi_rank

  subroutine c_macom_get_mpi_size(size) &
    bind(C, name="c_macom_get_mpi_size")
    integer(C_INT), intent(out) :: size
    size = mpi_procs
  end subroutine c_macom_get_mpi_size

  subroutine c_macom_finalize_mpi() &
    bind(C, name="c_macom_finalize_mpi")
    ! Empty - finalization handled by 4D-Var DA main
  end subroutine c_macom_finalize_mpi

  !-----------------------------------------------------------------------------
  ! II. Configuration (empty stubs)
  !-----------------------------------------------------------------------------
  subroutine c_macom_read_namelist() &
    bind(C, name="c_macom_read_namelist")
    ! Empty
  end subroutine c_macom_read_namelist

  subroutine c_get_macom_config_flags(mitice_flag, restart_flag, assim_flag, &
                                       init_iter, max_iter) &
    bind(C, name="c_get_macom_config_flags")
    logical(C_BOOL), intent(out) :: mitice_flag
    logical(C_BOOL), intent(out) :: restart_flag
    logical(C_BOOL), intent(out) :: assim_flag
    integer(C_INT), intent(out) :: init_iter
    integer(C_INT), intent(out) :: max_iter
    ! Empty - return default values
    mitice_flag = .false.
    restart_flag = .false.
    assim_flag = .false.
    init_iter = 0
    max_iter = 1
  end subroutine c_get_macom_config_flags

  !-----------------------------------------------------------------------------
  ! III. MACOM Model Operations (empty stubs)
  !-----------------------------------------------------------------------------
  subroutine c_macom_mpi_send_info_comp_to_io() &
    bind(C, name="c_macom_mpi_send_info_comp_to_io")
    ! Empty
  end subroutine c_macom_mpi_send_info_comp_to_io

  subroutine c_macom_misc_run_info_open() &
    bind(C, name="c_macom_misc_run_info_open")
    ! Empty
  end subroutine c_macom_misc_run_info_open

  subroutine c_macom_init_csp() &
    bind(C, name="c_macom_init_csp")
    ! Empty
  end subroutine c_macom_init_csp

  subroutine c_macom_restart_and_assim() &
    bind(C, name="c_macom_restart_and_assim")
    ! Empty
  end subroutine c_macom_restart_and_assim

  subroutine c_macom_run_csp_step() &
    bind(C, name="c_macom_run_csp_step")
    ! Empty
  end subroutine c_macom_run_csp_step

  subroutine c_macom_csp_io_main() &
    bind(C, name="c_macom_csp_io_main")
    ! Empty
  end subroutine c_macom_csp_io_main

  !-----------------------------------------------------------------------------
  ! IV. Sea Ice (Mitice) Functions (empty stubs for 4D-Var DA mode)
  !-----------------------------------------------------------------------------
  subroutine c_macom_initialize_mitice() &
    bind(C, name="c_macom_initialize_mitice")
    ! Empty - sea ice not used in 4D-Var DA mode
  end subroutine c_macom_initialize_mitice

  subroutine c_macom_mitice_init_all() &
    bind(C, name="c_macom_mitice_init_all")
    ! Empty - sea ice not used in 4D-Var DA mode
  end subroutine c_macom_mitice_init_all

  subroutine c_macom_finalize_mitice() &
    bind(C, name="c_macom_finalize_mitice")
    ! Empty - sea ice not used in 4D-Var DA mode
  end subroutine c_macom_finalize_mitice

  !-----------------------------------------------------------------------------
  ! V. 4D-Var DA Main Program (ONLY THIS ONE HAS REAL IMPLEMENTATION)
  !-----------------------------------------------------------------------------
  ! Interface to call the 4D-Var DA main program
  subroutine c_macom_4var_da_main() bind(C, name="c_macom_4var_da_main")
    implicit none

    ! This subroutine implements the complete program_main.f90 functionality
    ! but as a subroutine instead of a program

    integer :: IEXT, M, N, P, O, I, K
    real :: clg
    logical :: ln_assm_gain
    double precision, allocatable :: BB(:, :), VP(:, :), LM(:)

    ! Additional variables needed from program_main.f90
    integer :: NO, NP, NM, FLAG
    double precision :: MU
    double precision, allocatable :: XB(:), X0(:), TP(:), Delta_X(:), OM(:)
    double precision, allocatable :: X(:, :), Gradient(:), VV(:, :)
    double precision, allocatable :: OX(:), OY(:), OZ(:), OT(:), OS(:), OB(:)
    double precision, allocatable :: R(:), HX(:), Y(:, :), TY(:)
    double precision, allocatable :: ssh_gain(:), pbt_gain(:)
    double precision, allocatable :: tFld_gain(:, :), sFld_gain(:, :)
    double precision, allocatable :: uFld_gain(:, :), vFld_gain(:, :)
    integer, allocatable :: maskC_glo_ori(:, :), maskW_glo_ori(:, :), maskS_glo_ori(:, :)

    ! Initialize MPI (already done in C++)
    CALL mpi_process_init  ! must at first line
    macom_loop_i = 1
    macom_loop_end = 3

    if (mpi_rank == 0) then
      if (.not. allocated(XB)) allocate (XB(NP))
      if (.not. allocated(X0)) allocate (X0(NP))
      if (.not. allocated(TP)) allocate (TP(NP))
      if (.not. allocated(Delta_X)) allocate (Delta_X(NP))
      if (.not. allocated(OM)) allocate (OM(NM))
      if (.not. allocated(X)) allocate (X(NP, NM))
      if (.not. allocated(Gradient)) allocate (Gradient(NM))
      if (.not. allocated(VV)) allocate (VV(NM, NM))

      if (.not. allocated(BB)) allocate (BB(NM, NM))
      if (.not. allocated(VP)) allocate (VP(NM, NM))
      if (.not. allocated(LM)) allocate (LM(NM))

      open (12, FILE='../../ideal_observation/OBSERVATIONS_20200104.DAT')
      NO = 0
      do while (.true.)
        read (12, *, end=12)
        NO = NO + 1
      end do
12    continue
      close (12)
      if (.not. allocated(OX)) allocate (OX(NO))
      if (.not. allocated(OY)) allocate (OY(NO))
      if (.not. allocated(OZ)) allocate (OZ(NO))
      if (.not. allocated(OT)) allocate (OT(NO))
      if (.not. allocated(OS)) allocate (OS(NO))
      if (.not. allocated(OB)) allocate (OB(NO))
      if (.not. allocated(R)) allocate (R(NO))
      if (.not. allocated(HX)) allocate (HX(NO))
      if (.not. allocated(Y)) allocate (Y(NO, NM))
      if (.not. allocated(TY)) allocate (TY(NO))
      open (12, FILE='../../ideal_observation/OBSERVATIONS_20200104.DAT')
      do O = 1, NO
        read (12, *) OX(O), OY(O), OZ(O), OT(O), OB(O), R(O), OS(O)
      end do
      close (12)

      print *, NO

      open (12, FILE='./check/CHECK_OBS.DAT')
      do O = 1, NO
        write (12, *) OB(O), OS(O)
      end do
      close (12)

      open (97, FILE='../../X_MEMBER_HYCOM.DAT')
      print *, '--> reading X'
      do M = 1, NM
        do P = 1, NP
          read (97, *) X(P, M)
        end do
      end do
      close (97)

      open (8888, FILE='./check/COSTFUNCTION.TXT')
      open (9999, FILE='./check/OMEGA.TXT')
      open (6666, FILE='./check/HX_TMP.TXT')

      OM = 0.0D0
    end if

    ! Main computation loop - this is the core of program_main.f90
    do IEXT = 1, macom_loop_end

      if (mpi_rank == 0) print *, '--> This is for outloop:', IEXT

      FLAG = 0
      if (IEXT == macom_loop_end) FLAG = 1

      ! Call the main MACOM computation
      call nmefc_macom(.true.) ! ln_assm_gain = .true.

      ! Additional 4D-Var specific computations would go here
      ! (The rest of the program_main.f90 logic)

    end do

    if (mpi_rank == 0) then
      close (6666)
      close (8888)
      close (9999)
      if (allocated(XB)) deallocate (XB)
      if (allocated(X0)) deallocate (X0)
      if (allocated(TP)) deallocate (TP)
      if (allocated(Delta_X)) deallocate (Delta_X)
      if (allocated(OM)) deallocate (OM)
      if (allocated(X)) deallocate (X)
      if (allocated(Gradient)) deallocate (Gradient)
      if (allocated(VV)) deallocate (VV)
      if (allocated(OX)) deallocate (OX)
      if (allocated(OY)) deallocate (OY)
      if (allocated(OZ)) deallocate (OZ)
      if (allocated(OT)) deallocate (OT)
      if (allocated(OS)) deallocate (OS)
      if (allocated(OB)) deallocate (OB)
      if (allocated(R)) deallocate (R)
      if (allocated(HX)) deallocate (HX)
      if (allocated(Y)) deallocate (Y)
      if (allocated(TY)) deallocate (TY)
    end if

    ! Note: MPI_FINALIZE is not called here as it's handled by C++

  end subroutine c_macom_4var_da_main

  !-----------------------------------------------------------------------------
  ! Helper function
  !-----------------------------------------------------------------------------
  function c_int_to_string(i) result(str)
    integer, intent(in) :: i
    character(len=20) :: str
    write (str, '(I0)') i
  end function c_int_to_string

end module macom_fortran_wrapper
