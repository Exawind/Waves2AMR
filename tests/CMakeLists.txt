target_sources(${w2a_ocean_test_name}
    PRIVATE
    main_Ocean.cpp
    )

target_sources(${w2a_nwt_2d_test_name}
    PRIVATE
    main_NWT_2D.cpp
    )

target_sources(${w2a_nwt_3d_test_name}
    PRIVATE
    main_NWT_3D.cpp
    )

target_sources(${w2a_custom_test_name}
    PRIVATE
    test_modes_reading.cpp
    )