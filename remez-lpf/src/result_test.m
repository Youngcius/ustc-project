%��Ƽ���ָ��
clear;
freq = [0.3, 0.5];
mag = [1, 0];
alpha_p = 1; %dB
alpha_s = 40; % dB

% ����Ȩ��������
delta_1 = (10^(abs(alpha_p)/20) - 1) / (10^(abs(alpha_p)/20) + 1);
delta_2 = 10^(-abs(alpha_s)/20);
rip = [delta_1, delta_2];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% my_reme��Ʒ������
hn = my_remez(freq, rip);
N = length(hn);

figure;
stem(0:N-1,hn);
title('�˲����弤��Ӧ��������ʵ�֣�');

figure;
freqz(hn);
title('�˲���Ƶ�����ԣ���ʵ�֣�');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matlab�Դ�remez��Ʒ������

[N, f0, m0, w] = remezord(freq, mag, rip);
hn = remez(N, f0, m0, w);
n = 0:N;

figure;
stem(n, hn);
title('�˲����弤��Ӧ������Matlab���ã�');

figure;
freqz(hn);
title('�˲���Ƶ�����ԣ�Matlab���ã�');

% kaiser window

N = 15;
window = kaiser(N,3.4);
b = fir1(N-1,0.4,window);
figure;
freqz(b,1)
title('Kaiser��15���˲���')


% iir��Բ�˲���
% FDATool

% ���ɴ�����������
fp = [0.1, 0.2, 0.3 ,0.4, 0.5, 0.6];
fs = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7];
f0 = [zeros(length(fp),1),fp',fs',ones(length(fp),1)];

figure;
passbands = fs - fp;
deltas = ones(length(passbands), 1);
for ii = 1:length(fp)
    [hn, delta] = remez(N, f0(ii,:), m0);
    deltas(ii) = delta;
    figure;
    freqz(hn);
    title(['passband: ',num2str(0),' - ',num2str(f0(ii,2)),', stopband: ',num2str(f0(ii,3)),' - ',num2str(1)]);
end

% ת����ϵ
fp = linspace(0.1,0.38,20);
fp = sort(fp, 'descend');
fs = linspace(0.42, 0.7, 20);
f0 = [zeros(length(fp),1),fp',fs',ones(length(fp),1)];
passbands = fs - fp;
deltas = ones(length(passbands), 1);
for ii = 1:20
    [hn, delta] = remez(N, f0(ii,:), m0);
    deltas(ii) = delta;
end

figure;
% subplot(2,1,1)
plot(passbands,deltas)
xlabel('passband');
ylabel('delta');
title('����-��������ϵ��N=15)');

% subplot(2,1,2);
% plot(passbands, 20*los10(1-deltas));


% [N, f0, m0, w] = remezord(freq, mag, rip);
% hn = remez(N, f0, m0, w);
% n = 0:N;
% % f0 = [0, 0.3, 0.5, 1];
% m0 = [1, 1, 0, 0];


% for ii = 1:length(wp)
%    hn = remez(N, f0,
%    hn = my_remez([wp(ii),ws(ii)], rip);
%    subplot(3,
% end


% ��ͬ���ɴ���

f0 = [0.3, 0.5];
alpha_p = [1, 1, 1, 0.6, 0.8, 1];
alpha_s = [40, 60, 80,40, 40, 40];
delta_1 = (10.^(abs(alpha_p)/20) - 1) ./ (10.^(abs(alpha_p)/20) + 1);
delta_2 = 10.^(-abs(alpha_s)/20);
rip = [delta_1', delta_2'];
% passbands = fs - fp;
% deltas = ones(length(passbands), 1);
figure;
for ii = 1:6
    deltas = rip(ii,:);
    hn = my_remez(f0, deltas);
    [h,w] = freqz(hn);
    
    N = length(hn);
    if ii==1
        subplot(3,2,1);
    elseif ii==2
        subplot(3,2,3);
    elseif ii==3
        subplot(3,2,5);
    elseif ii==4
        subplot(3,2,2);
    elseif ii==5
        subplot(3,2,4);
    else
        subplot(3,2,6);
    end
                       
%     if rem(ii,2)==0
%         subplot(3,2,(ii-3)*2);
%     else
%         subplot(3,2,ii*2-1)
%     end
    plot(w/pi, 20*log10(abs(h)));
    legend([num2str(N),'�ף�ͨ��',num2str(alpha_p(ii)),'dB�����',num2str(alpha_s(ii)),'dB']);
    grid;
%     if ii == 6
%         title('��ͬ���ɴ���0.3-0.5��ͬ�������ָ��ķ�Ƶ����');
%     end
%     deltas(ii) = delta;
end

