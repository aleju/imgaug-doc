import numpy as np
# import pyfftw as fftw


# Provide an augmented image example from some random draws
# function augmented_image = augment_sar( complex_image, support_data )
def augment_sar(complex_image, support_data):
       
    additive_noise_sigma = 0.05, 
    multiplicative_noise_factor = 1
    shift_size = 2.0                                  #Doesn't matter with new comments
    wrap_noise = 30
    magnitude_noise_factor = 15              
    
    # not_rand = 0.77 # for debug


    spacing = np.minimum(support_data["line_spacing"], support_data["sample_spacing"])
    shift_noise = shift_size/spacing  # pixels = m/(m/pixel)
    
    
    nx = complex_image.shape[0]
    ny = complex_image.shape[1]

    x = np.arange(1, nx+1, dtype='int32')
    y = np.arange(1, ny+1, dtype='int32')
    h_xval, h_yval = np.meshgrid(x, y, sparse=False, indexing='ij')

    shift_x = np.random.randn(1)*shift_noise  # az
    shift_y = np.random.randn(1)*shift_noise  # range
    
    # shift_x = not_rand*shift_noise  # az
    # shift_y = not_rand*shift_noise  # range

    z = shiftimage(complex_image, shift_x, shift_y)

    log_mult_noise_fac = np.log(multiplicative_noise_factor)

    multiplicative_factor = np.exp( (2*np.random.rand(nx, ny)-1)*log_mult_noise_fac )
    # multiplicative_factor = np.exp( (2*not_rand*np.ones((nx, ny))-1)*log_mult_noise_fac )

    additive_noise = (np.random.randn(nx, ny)+np.random.randn(nx, ny)*1j)*additive_noise_sigma
    # additive_noise = (not_rand*np.ones((ny, nx))+not_rand*np.ones((nx, ny))*1j)*additive_noise_sigma

    # band limit the noise
    wt = mkwt('MSTAR', complex_image.shape[0])  # find Taylor window for MSTAR type data, util

    N = np.fft.fft2(additive_noise)

    wt = wt[0:N.shape[0],0:N.shape[1]]
    # print("z.shape ",z.shape)
    # print("N.shape ",N.shape)
    # print("wt.shape ",wt.shape)
    N = N*wt

    additive_noise = np.fft.ifft2(N)

    multiplicative_factor = np.fft.ifft2(wt*np.fft.fft2(multiplicative_factor))

    mval = np.mean(np.abs(multiplicative_factor))

    multiplicative_factor = multiplicative_factor/mval

    z = z*multiplicative_factor + additive_noise

    # phase ramps
    wrap_x = (2*np.random.rand(1)-1)*wrap_noise
    phi0x = np.random.rand(1)*2*np.pi
    wrap_y = (2*np.random.rand(1)-1)*wrap_noise
    phi0y = np.random.rand(1)*2*np.pi
    # wrap_x = (2*not_rand-1)*wrap_noise
    # phi0x = not_rand*2*np.pi
    # wrap_y = (2*not_rand-1)* wrap_noise
    # phi0y = not_rand* 2 * np.pi

    # watch out below
    phaseramp = np.exp((h_xval*1j*2*np.pi*wrap_x/ny)-1j*phi0x)*np.exp((h_yval*1j*2*np.pi*wrap_y/nx)-1j*phi0y)

    z = z*phaseramp

    magnitude_factor = np.exp( (2*np.random.rand(1)-1)*np.log(magnitude_noise_factor) )
    # magnitude_factor = np.exp( (2*not_rand-1)*np.log(magnitude_noise_factor) )

    overall_phase = np.random.rand(1)*2*np.pi
    # overall_phase = not_rand*2*np.pi

    return z*magnitude_factor*np.exp(1j*overall_phase),shift_x,shift_y


# shift an image
def shiftimage(complex_image, da, dr):

    na = complex_image.shape[0]
    nr = complex_image.shape[1]
    
    r_shift = np.exp(-1j * 2 * np.pi * dr * np.hstack((np.r_[0.:np.floor(nr/2.)], np.r_[np.floor(-nr/2):0])) / nr)
    r_shift.shape = (1, nr)
    
    a_shift = np.exp(-1j * 2 * np.pi * da * np.hstack((np.r_[0.:np.floor(na/2.)], np.r_[np.floor(-na/2):0])) / na)
    a_shift.shape = (na, 1)
    
    # F = fftw.interfaces.numpy_fft.fft2(complex_image)
    F = np.fft.fft2(complex_image)
    C = F*np.matmul(a_shift, r_shift)
    # shifted_image = fftw.interfaces.numpy_fft.ifft2(C)
    shifted_image = (np.fft.ifft2(C))

    return shifted_image
    # return shifted_image, F, c1, C


# make weighting window (Taylor window -35 dB) for injecting into MSTAR
def mkwt(casename,n1):

    wt = []

    if casename == "MSTAR":
        # MSTAR uses a -35 dB Taylor window filter for the weights
        # ntay = 126 # support  # ORIGINAL
        # n1 = 158 # image size # ORIGINAL (now input)
        # ntay = int(np.ceil(0.8*n1))
        ntay = np.int32(0.8 * n1)

        # y = np.fft.ifftshift(taylorwin(ntay,5,-35))
        y = np.fft.ifftshift(taylorwin(ntay, 5, -35))                     #util

        # y = [y(1:(ntay/2)); zeros((n1-ntay),1); y(ntay/2+(1:(ntay/2)))];
        a = y[  np.int32(np.r_[0:(ntay/2)])  ]
        b = np.zeros((n1-ntay,1))
        c = y[  np.int32(np.r_[0:(ntay/2)]+(ntay/2))  ]
        y = np.vstack((a,b,c))

        # w = y*y'
        w = np.matmul(y,np.conj(np.transpose(y)))

        # wt=w/sqrt(sum(sum(conj(ifft2(w)).*ifft2(w)))) # power preserving ?
        wt=w/np.sqrt(np.sum(  np.conj(np.fft.ifft2(w))  *  np.fft.ifft2(w)  )) # power preserving ?
    else:
        print("function mkwt does not yet know about " + casename)

    return wt


# Taylor window.
# output:       N-point Taylor window
# input nbar:   nearly constant-level sidelobes adjacent to the mainlobe.
#               nbar must be an integer greater than or equal to one.
# input sll:    sll maximum sidelobe level in dB relative to the mainlobe peak.
#               sll must be negative
# nbar should satisfy nbar >= 2*A^2+0.5, where A is equal to
# acosh(10^(-sll/20))/pi, otherwise the sidelobe level specified is not guaranteed
def taylorwin(n, nbar, sll):

    # A = acosh(10^(-SLL/20))/pi;
    A = np.arccosh(10**(-sll/20))/np.pi

    # Taylor pulse widening (dilation) factor.
    # sp2 = NBAR^2/(A^2 + (NBAR-.5)^2);
    sp2 = nbar**2/(A**2 + (nbar-0.5)**2)

    # w = ones(N,1);
    w = np.ones((n,1))

    # Fm = zeros(NBAR-1,1);
    Fm = np.zeros((nbar-1,1))

    # summation = 0;
    summation = 0

    # k = (0:N-1)';
    k = np.r_[0:n]
    k.shape = (n, 1)

    # xi = (k-0.5*N+0.5)/N;
    xi = (k-0.5*n+0.5)/n

    # for m = 1:(NBAR-1),
    #     Fm(m) = calculateFm(m,sp2,A,NBAR);
    #     summation = Fm(m)*cos(2*pi*m*xi)+summation;
    # end
    for m in np.r_[1:nbar]:
        Fm[m-1] = calculateFm(m, sp2, A, nbar)                                #Util
        summation = Fm[m-1] *np.cos(2*np.pi*m*xi)+summation

    # w = w + 2*summation;
    return w + 2*summation

# Calculate the cosine weights.
def calculateFm(m,sp2,A,nbar):

    # n = (1:NBAR-1)';
    n = np.r_[1:nbar]
    n.shape = (nbar-1, 1)

    # p = [1:m-1, m+1:NBAR-1]'; % p~=m
    p = np.hstack( ( np.r_[1:m] , np.r_[m+1:nbar] ) )
    p.shape = (p.shape[0],1)

    # Num = prod(  (1 - (m^2/sp2)./(A^2+(n-0.5).^2))  );
    a = (m**2/sp2)
    b = (A**2 + (n-0.5)**2)
    c = 1 - (m**2/sp2) / (A**2 + (n-0.5)**2)
    Num = np.prod(1 - (m**2/sp2) / (A**2 + (n-0.5)**2)  )
    # Den = prod((1 - m^2./p.^2));
    Den = np.prod((1 - m**2./p**2))

    # Fm = ((-1)^(m+1).*Num)./(2.*Den);
    return ((-1)**(m+1)*Num)/(2*Den)
