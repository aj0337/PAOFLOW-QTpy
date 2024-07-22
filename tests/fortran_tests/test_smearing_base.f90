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
END MODULE smearing_base_module


PROGRAM SmearingTest


   USE smearing_base_module

   IMPLICIT NONE

   INTEGER, PARAMETER :: dbl = selected_real_kind(14,200)
   REAL(dbl) :: x
   CHARACTER(50) :: smearing_type

   x = 0.5
   smearing_type = "lorentzian"
   PRINT *, "Smearing Function (", TRIM(smearing_type), "): ", smearing_func(x, smearing_type)

   smearing_type = "gaussian"
   PRINT *, "Smearing Function (", TRIM(smearing_type), "): ", smearing_func(x, smearing_type)

   smearing_type = "fermi-dirac"
   PRINT *, "Smearing Function (", TRIM(smearing_type), "): ", smearing_func(x, smearing_type)

   smearing_type = "methfessel-paxton"
   PRINT *, "Smearing Function (", TRIM(smearing_type), "): ", smearing_func(x, smearing_type)

   smearing_type = "marzari-vanderbilt"
   PRINT *, "Smearing Function (", TRIM(smearing_type), "): ", smearing_func(x, smearing_type)

END PROGRAM SmearingTest
