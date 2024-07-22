MODULE smearing_base_module

   IMPLICIT NONE
   INTEGER, PARAMETER :: dbl = selected_real_kind(14,200)
   REAL(dbl), PARAMETER ::     ONE = 1.0_dbl
   REAL(dbl), PARAMETER ::     TWO = 2.0_dbl
   REAL(dbl), PARAMETER ::      PI = 3.14159265358979323846_dbl
   REAL(dbl), PARAMETER ::  SQRTPI = 1.77245385090551602729_dbl
   REAL(dbl), PARAMETER ::   SQRT2 = 1.41421356237309504880_dbl

   PRIVATE

   PUBLIC :: smearing_func

CONTAINS

   FUNCTION smearing_func(x, smearing_type )
      IMPLICIT NONE
      REAL(dbl)          :: smearing_func
      REAL(dbl)          :: x
      CHARACTER(*)       :: smearing_type
      CHARACTER(13)      :: subname="smearing_func"
      REAL(dbl)          :: cost

      smearing_func = ONE

      SELECT CASE (TRIM(smearing_type))
       CASE ( "lorentzian" )
         cost = ONE / PI
         smearing_func = cost * ONE/( ONE + x**2 )

       CASE ( "gaussian" )
         cost = ONE / SQRTPI
         smearing_func = cost * EXP( -x**2 )

       CASE ( "fermi-dirac", "fd" )
         cost = ONE / TWO
         smearing_func = cost * ONE / ( ONE + COSH(x) )

       CASE ( "methfessel-paxton", "mp" )
         cost = ONE / SQRTPI
         smearing_func = cost * EXP( -x**2 ) * ( 3.0_dbl/2.0_dbl - x**2 )

       CASE ( "marzari-vanderbilt", "mv" )
         cost = ONE / SQRTPI
         smearing_func = cost * EXP( -(x- ONE/SQRT2 )**2 ) * ( TWO - SQRT2 * x )

         !  CASE DEFAULT
         !    CALL errore(subname, 'invalid smearing_type = '//TRIM(smearing_type),1)
      END SELECT
   END FUNCTION smearing_func

!**********************************************************
   SUBROUTINE smearing_init()
      !**********************************************************
      IMPLICIT NONE
      CHARACTER(13)          :: subname="smearing_init"
      REAL(dbl)              :: cost, x
      INTEGER                :: i, ierr
      !
      INTEGER                   :: is_start, is_end    ! index for eps_s grid
      REAL(dbl)                 :: eps_sx              ! eps_s grid extrema
      !
      INTEGER                   :: ip_start, ip_end    ! index for eps_p grid
      REAL(dbl)                 :: eps_px              ! eps_p grid extrema
      !
      INTEGER                   :: nfft                ! dim of the FFT grid
      REAL(dbl)                 :: Tmax                ! FFT grid extrema
      REAL(dbl), ALLOCATABLE    :: fft_grid(:)         ! actual smearing funKtion
      COMPLEX(dbl), ALLOCATABLE :: auxs_in(:)          ! cmplx smear for FFT
      COMPLEX(dbl), ALLOCATABLE :: auxp_in(:)          ! pole for FFT
      COMPLEX(dbl), ALLOCATABLE :: auxs_out(:)         ! FFT output for aux1
      COMPLEX(dbl), ALLOCATABLE :: auxp_out(:)         ! FFT input for aux2
      COMPLEX(dbl), ALLOCATABLE :: wrapped(:)          ! auxiliary vect
      !
      INTEGER                   :: ix_start, ix_end    ! index for g_smear grid


      CALL timing ( 'smearing_init', OPR='start')
      CALL log_push ( 'smearing_init' )
      !
      ! few checks
      IF ( alloc ) CALL errore(subname,'smearing already allocated',1)
      IF ( delta_ratio < ZERO  ) CALL errore(subname,'delta_ratio too small',1)
      IF ( delta_ratio > EPS_m1) CALL errore(subname,'delta_ratio too large',1)

      !
      ! define the xgrid
      nx = 2 * INT( TWO * xmax / delta_ratio )
      dx = TWO * xmax / REAL(nx, dbl)
      !
      ALLOCATE( g_smear(nx), xgrid(nx), STAT=ierr )
      IF ( ierr /=0 ) CALL errore(subname,'allocating g_smear, xgrid',ABS(ierr))
      !
      DO i = 1, nx
         !
         xgrid(i) = -REAL(nx/2,dbl)*dx + REAL(i-1, dbl) * dx
         !
      ENDDO

      !
      ! define the fft grid
      !
      ! eps_px (pole)   = xmax +   eps_sx
      ! Tmax (FFT extr) = xmax + 2*eps_sx
      !

      ! define eps_sx  (half of the width of the smearing function)
      eps_sx = 15.0_dbl
      !
      eps_px = xmax + eps_sx
      Tmax   = xmax + TWO * eps_sx
      !
      nfft   = 1+ INT ( ( Tmax / xmax ) * nx )
      !
      ! find a "good" fft dimension (machine dependent)
      !
      nfft = good_fft_order_1dz( nfft )
      !
      !
      ALLOCATE( fft_grid( nfft ), STAT=ierr )
      IF (ierr/=0) CALL errore(subname, 'allocating fft_grid',ABS(ierr))
      !
      ALLOCATE( auxs_in( nfft ), auxp_in( nfft ), STAT=ierr )
      IF (ierr/=0) CALL errore(subname, 'allocating auxs_in, auxp_in',ABS(ierr))
      !
      ALLOCATE( auxs_out( nfft ), auxp_out( nfft ), STAT=ierr )
      IF (ierr/=0) CALL errore(subname, 'allocating auxs_out, auxp_out',ABS(ierr))
      !
      ALLOCATE( wrapped( nfft ), STAT=ierr )
      IF (ierr/=0) CALL errore(subname, 'allocating wrapped',ABS(ierr))
      !
      alloc = .true.
      !
      DO i=1,nfft
         !
         fft_grid(i) = -REAL(nfft/2, dbl)*dx + REAL(i-1,dbl) *dx
         !
      ENDDO

      !
      ! find the extrema of interest on the fft_grid
      CALL locate( fft_grid, nfft, -eps_sx, is_start)
      CALL locate( fft_grid, nfft,  eps_sx, is_end)
      !
      CALL locate( fft_grid, nfft, -eps_px, ip_start)
      CALL locate( fft_grid, nfft,  eps_px, ip_end)

      !
      ! define the smearing function
      !
      auxs_in(:)  = CZERO
      auxs_out(:) = CZERO

      cost = ONE / delta
      !
      DO i= is_start, is_end
         !
         x          = fft_grid(i)
         auxs_in(i) = cost * smearing_func( x, TRIM(smearing_type) )
         !
      ENDDO

      !
      ! define the input pole function
      !
      auxp_in(:)  = CZERO
      auxp_out(:) = CZERO
      !
      cost = ONE
      DO i = ip_start, ip_end
         !
         x          = fft_grid(i)
         auxp_in(i) = cost / ( x + CI * delta_ratio )
         !
      ENDDO

      !
      ! perform the FFT
      !
      !
      ! perform the smearing func wrapping
      !
      CALL locate( fft_grid, nfft, ZERO, i)
      IF ( fft_grid(i) < ZERO ) i = i+1
      !
      wrapped(:) = CSHIFT( auxs_in(:), i-1 )
      auxs_in(:) = wrapped(:)


      !
      ! freq to time FT
      !
      CALL timing ( 'cft_1z', OPR='start')
      CALL log_push ( 'cft_1z')
      !
      CALL cft_1z ( auxs_in, 1, nfft, nfft, -1, auxs_out)
      CALL cft_1z ( auxp_in, 1, nfft, nfft, -1, auxp_out)
      !
      CALL timing ( 'cft_1z', OPR='stop')
      CALL log_pop ( 'cft_1z')

      !
      ! perform the convolution
      !
      cost = TWO * Tmax
      DO i=1, nfft
         !
         auxp_out(i) = cost * auxp_out(i) * auxs_out(i)
         !
      ENDDO

      !
      ! backwards fft
      !
      CALL timing ( 'cft_1z', OPR='start')
      CALL log_push ( 'cft_1z')
      !
      CALL cft_1z ( auxp_out, 1, nfft, nfft, 1, auxp_in)
      !
      CALL timing ( 'cft_1z', OPR='stop')
      CALL log_pop ( 'cft_1z')

      !
      ! smeared green function extraction
      !
      CALL locate( fft_grid, nfft, -xmax, ix_start )
      ix_end = ix_start + nx -1
      !
      g_smear(:) = auxp_in(ix_start:ix_end)

      !
      ! local cleaning
      !
      DEALLOCATE( fft_grid, STAT=ierr )
      IF (ierr/=0) CALL errore(subname, 'deallocating fft_grid',ABS(ierr))
      !
      DEALLOCATE( auxs_in, auxp_in, STAT=ierr )
      IF (ierr/=0) CALL errore(subname, 'deallocating auxs_in, auxp_in',ABS(ierr))
      !
      DEALLOCATE( auxs_out, auxp_out, STAT=ierr )
      IF (ierr/=0) CALL errore(subname, 'deallocating auxs_out, auxp_out',ABS(ierr))
      !
      DEALLOCATE( wrapped, STAT=ierr )
      IF (ierr/=0) CALL errore(subname, 'deallocating wrapped',ABS(ierr))

      CALL timing ( 'smearing_init', OPR='stop')
      CALL log_pop ( 'smearing_init')

   END SUBROUTINE smearing_init

END MODULE smearing_base_module

PROGRAM SmearingInitTest
   USE smearing_base_module

   IMPLICIT NONE
   INTEGER, PARAMETER :: dbl = selected_real_kind(14,200)

   ! Test parameters
   REAL(dbl), PARAMETER :: xmax = 10.0_dbl
   REAL(dbl), PARAMETER :: delta = 0.1_dbl
   REAL(dbl), PARAMETER :: delta_ratio = 0.01_dbl
   CHARACTER(13), PARAMETER :: smearing_type = "lorentzian"

   ! Call the initialization subroutine
   CALL smearing_init()

   ! Print some results or perform further tests if needed
   PRINT *, "Smearing initialization completed successfully."

END PROGRAM SmearingInitTest
